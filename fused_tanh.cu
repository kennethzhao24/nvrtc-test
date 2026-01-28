// fused_tanh.cu
// Buildable as a PyTorch CUDA extension. Provides fused_forward() that applies y = tanh(x) elementwise.
// PyTorch's CUDA kernel dispatch system is used to select the appropriate device implementation (whether it's built-in or not).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Forward kernel: y = tanh(x)
template <typename scalar_t>
__global__ void tanh_forward_kernel(const scalar_t* __restrict__ x,
                                   scalar_t* __restrict__ y,
                                   int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        scalar_t vx = x[i];
        scalar_t t = tanh(vx);
        y[i] = t;
    }
}

// Launch helper
static inline void launch_tanh_forward(const at::Tensor& x_contig, at::Tensor& y_contig) {
    const int64_t N = x_contig.numel();
    if (N == 0) return;

    const int threads = 256;
    // Heuristic: up to 32 blocks per SM
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    int sm_count = prop->multiProcessorCount;
    int max_blocks = sm_count * 32;
    int64_t blocks_needed = (N + threads - 1) / threads;
    int blocks = static_cast<int>(std::min<int64_t>(blocks_needed, std::max<int>(1, max_blocks)));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "tanh_forward_kernel", [&] {
        const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
        scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
        tanh_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(x_ptr, y_ptr, N);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// C++/CUDA binding
at::Tensor fused_forward(const at::Tensor& tensor_0) {
    TORCH_CHECK(tensor_0.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(tensor_0.is_floating_point(), "Input tensor must be floating point.");

    // Ensure contiguous layout for efficient access
    at::Tensor x_contig = tensor_0.contiguous();
    at::ScalarType dtype = x_contig.scalar_type();

    // Allocate output tensor with same shape and dtype
    at::Tensor y_contig = at::empty_like(x_contig);

    // Launch fused forward kernel
    launch_tanh_forward(x_contig, y_contig);

    // Return as a single-element list to match the Python function's signature [tensor_1]
    return { y_contig };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Fused operator forward (CUDA): y = tanh(x)");
}
