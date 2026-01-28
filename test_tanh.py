import os
import time
import torch
from torch.utils.cpp_extension import load

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6' # A40 GPU

def test():
    print(f"--- Verifying Kernel ---")
    verify_start = time.time()
    # Create input
    x = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
    
    # 2. Get GPU capability (Important for your A40)
    capability = torch.cuda.get_device_capability()
    arch_flag = f"-arch=sm_{capability[0]}{capability[1]}"

    # 3. JIT Compile the extension
    print(f"--- Compiling Kernel (JIT) ---")
    comp_start = time.time()

    fused_tanh_lib = load(
        name="fused_tanh_extension",
        sources=["fused_tanh.cu"],
        extra_cuda_cflags=[
            arch_flag,
            "-O3",
            "--use_fast_math",
            "-std=c++17"
        ],
        verbose=True # Set to True to see compilation logs
    )

    comp_end = time.time()
    print(f"Compilation Time: {comp_end - comp_start:.4f} seconds")

    # Call your C++ function
    # Note: Your C++ code returns a single Tensor, so we capture it directly
    out_custom = fused_tanh_lib.fused_forward(x)
    
    # Compare with PyTorch native Tanh
    out_pytorch = torch.tanh(x)
    
    if torch.allclose(out_custom, out_pytorch, atol=1e-5):
        print("✅ Success: Custom Tanh matches PyTorch!")
    else:
        print("❌ Error: Results do not match.")

    verify_end = time.time()
    print(f"Verifiction Time: {verify_end - verify_start:.4f} seconds")

if __name__ == "__main__":
    test()