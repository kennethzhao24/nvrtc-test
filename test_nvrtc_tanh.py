import time
import torch
import cupy as cp

# 1. Define the Kernel Template
# Note: We use double braces {{ }} for C++ and single braces {dtype} for Python.
# We also add extern "C" so NVRTC doesn't "mangle" the function name.
KERNEL_TEMPLATE = r'''
extern "C" {{
    __global__ void tanh_forward_kernel(const {dtype}* __restrict__ x,
                                       {dtype}* __restrict__ y,
                                       long long N) {{
        long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        long long stride = blockDim.x * gridDim.x;
        for (long long i = idx; i < N; i += stride) {{
            {dtype} vx = x[i];
            y[i] = ({dtype})tanh((float)vx);
        }}
    }}
}}
'''

def run_nvrtc_tanh(x_torch):
    # Ensure tensor is contiguous on the GPU
    x_torch = x_torch.contiguous()
    out_torch = torch.empty_like(x_torch)
    N = x_torch.numel()

    # Map Torch dtype to C++ type
    dtype_str = 'float' if x_torch.dtype == torch.float32 else 'half'
    
    # 2. Compile with NVRTC via CuPy
    # We pass only the C++ code to the compiler
    cuda_source = KERNEL_TEMPLATE.format(dtype=dtype_str)

    print(f"--- Compiling Kernel (NVCC) ---")
    comp_start = time.time()

    options = ("--use_fast_math", "-std=c++17")
    module = cp.RawModule(code=cuda_source, options=options, backend='nvrtc')
    kernel = module.get_function("tanh_forward_kernel")
    comp_end = time.time()
    print(f"Compilation Time: {comp_end - comp_start:.4f} seconds")

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

    return out_torch

# --- Verification Section ---
def verify():

    print(f"--- Verifying Kernel ---")

    verify_start = time.time()
    
    # Create test data
    x = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
    
    # Run custom kernel
    y_custom = run_nvrtc_tanh(x)
    
    # Run PyTorch native
    y_torch = torch.tanh(x)
    
    # Check results
    if torch.allclose(y_custom, y_torch, atol=1e-5):
        print("✅ Verification Successful! Custom Tanh matches PyTorch.")
    else:
        print("❌ Verification Failed! Discrepancy detected.")
        print(f"Max Diff: {(y_custom - y_torch).abs().max()}")

    verify_end = time.time()
    print(f"Verifiction Time: {verify_end - verify_start:.4f} seconds")

if __name__ == "__main__":
    verify()