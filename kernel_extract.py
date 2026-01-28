import torch
import cupy as cp
import re
import os

class AutoCudaLoader:
    def __init__(self, directory="."):
        self.directory = directory
        self.kernels = {}  # Stores raw strings
        self.modules = {}  # Stores compiled cp.RawModule
        self.compile_options = ("--use_fast_math", "-std=c++17")

    def _extract_kernel_from_file(self, file_path):
            with open(file_path, 'r') as f:
                content = f.read()

            # Regex: Find any __global__ function and its name
            pattern = r"__global__\s+void\s+(\w+)\s*\(([\s\S]+?)\)\s*\{"
            matches = re.finditer(pattern, content)
            
            extracted = []
            for match in matches:
                kernel_name = match.group(1)
                
                # Use brace counting to find the end of the function body
                start_idx = match.start()
                brace_count = 0
                end_idx = -1
                for i in range(start_idx, len(content)):
                    if content[i] == '{': brace_count += 1
                    elif content[i] == '}': 
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                full_body = content[start_idx:end_idx]
                
                # IMPORTANT: Do NOT use .replace("{", "{{") anymore.
                # We keep the raw C++ code exactly as it is.
                extracted.append((kernel_name, full_body))
            return extracted

    def load_all(self):
        """Crawls directory and extracts all kernels."""
        for file in os.listdir(self.directory):
            if file.endswith(".cu"):
                path = os.path.join(self.directory, file)
                kernels = self._extract_kernel_from_file(path)
                for name, body in kernels:
                    # Wrap in extern "C" for NVRTC
                    self.kernels[name] = f'extern "C" {{\n{body}\n}}'
        print(f"✅ Loaded {len(self.kernels)} kernels: {list(self.kernels.keys())}")

    def get_kernel(self, kernel_name, dtype_torch):
            """Compiles (if needed) and returns the kernel."""
            # Handle dtype mapping
            if dtype_torch == torch.float32 or dtype_torch == 'float':
                dtype_str = 'float'
            elif dtype_torch == torch.float16 or dtype_torch == 'half':
                dtype_str = 'half'
            elif dtype_torch == torch.bfloat16 or dtype_torch == 'bfloat16':
                dtype_str = 'nv_bfloat16'
            else:
                dtype_str = 'float'

            cache_key = f"{kernel_name}_{dtype_str}"

            if cache_key not in self.modules:
                # 1. Provide manual definitions and fix CUDA 12.8 header issues
                header_str = """
                #ifndef __CUDACC_RTC__
                #define __CUDACC_RTC__
                #endif
                
                // Fix for CUDA 12.8 BF16 header issues
                #define NV_IF_ELSE_TARGET(target, if_clause, else_clause) if_clause
                #define NV_IS_DEVICE 1
                
                typedef signed char        int8_t;
                typedef short              int16_t;
                typedef int                int32_t;
                typedef long long          int64_t;
                typedef unsigned char      uint8_t;
                typedef unsigned short     uint16_t;
                typedef unsigned int       uint32_t;
                typedef unsigned long long uint64_t;
                
                #include <cuda_fp16.h>
                """
                
                # Only include BF16 if we are actually using it to avoid 12.8 errors
                if dtype_str == 'nv_bfloat16':
                    header_str += "\n#include <cuda_bf16.h>"

                # 2. Extract and replace scalar_t (same as before)
                raw_source = self.kernels[kernel_name]
                source = re.sub(r'\bscalar_t\b', dtype_str, raw_source)
                
                # 3. Flatten extern "C"
                source = source.replace('extern "C" {', '').rstrip().rstrip('}')
                
                # 4. Final Assembly
                final_source = f"{header_str}\n\nextern \"C\" {{\n{source}\n}}"
                
                # 5. Important: Add the specific include path for HPC SDK
                # Since your logs show a very specific path, let's point NVRTC there
                hpc_include = "-I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/include"
                options = self.compile_options + (hpc_include,)
                
                try:
                    module = cp.RawModule(code=final_source, options=options, backend='nvrtc')
                    self.modules[cache_key] = module.get_function(kernel_name)
                except cp.cuda.compiler.CompileException as e:
                    # ... (debug print)
                    raise e
            return self.modules[cache_key]

def verify_kernel(kernel_name):
    kernel = loader.get_kernel(kernel_name, 'float')
    return kernel

# --- Usage Example ---
loader = AutoCudaLoader(directory="/u/yzhao25/nvrtc_tests")
loader.load_all()

kernel = verify_kernel("tanh_forward_kernel")

x_torch = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)

# Ensure tensor is contiguous on the GPU
x_torch = x_torch.contiguous()
out_torch = torch.empty_like(x_torch)
N = x_torch.numel()

# 3. Configure Launch Params
threads = 256
# Grid-stride loop: blocks can be smaller than total N
blocks = min((N + threads - 1) // threads, 65535)

# 4. Launch using PyTorch data pointers
kernel(
    grid=(blocks,), 
    block=(threads,), 
    args=(cp.asarray(x_torch), cp.asarray(out_torch), N)
)


# --- Verification Section ---
y_torch = torch.tanh(x_torch)

# Check results
if torch.allclose(out_torch, y_torch, atol=1e-5):
    print("✅ Verification Successful! Custom Tanh matches PyTorch.")
else:
    print("❌ Verification Failed! Discrepancy detected.")
    print(f"Max Diff: {(out_torch - y_torch).abs().max()}")