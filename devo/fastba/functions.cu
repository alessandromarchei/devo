#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>




void print_tensor_stats(const torch::Tensor& tensor, const std::string& name) {
  std::cout << name << " - Size: " << tensor.sizes() 
            << ", Mean : " << tensor.mean().item<float>()
            << ", Std: " << tensor.std().item<float>()
            << ", Min: " << tensor.min().item<float>()
            << ", Max: " << tensor.max().item<float>() << std::endl;
}

__device__ void print_accessor_stats(const at::GenericPackedTensorAccessor<float, 1, at::RestrictPtrTraits, int32_t>& accessor, const char* name) {
    float sum = 0.0f;
    float min_v = 1e10f;
    float max_v = -1e10f;
    int count = accessor.size(0);

    for (int i = 0; i < accessor.size(0); i++) {
        float val = accessor[i];
        sum += val;
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }

    float mean = sum / (count > 0 ? count : 1);
    printf("%s -> Size: [%d], Mean: %.4f, Min: %.4f, Max: %.4f\n",
           name, accessor.size(0), mean, min_v, max_v);
}

__device__ void print_accessor_stats(const at::GenericPackedTensorAccessor<float, 2, at::RestrictPtrTraits, int32_t>& accessor, const char* name) {
    float sum = 0.0f;
    float min_v = 1e10f;
    float max_v = -1e10f;
    int count = accessor.size(0) * accessor.size(1);

    for (int i = 0; i < accessor.size(0); i++) {
        for (int j = 0; j < accessor.size(1); j++) {
            float val = accessor[i][j];
            sum += val;
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
        }
    }

    float mean = sum / (count > 0 ? count : 1);
    printf("%s -> Size: [%d, %d], Mean: %.4f, Min: %.4f, Max: %.4f\n",
           name, accessor.size(0), accessor.size(1), mean, min_v, max_v);
}

__device__ void print_accessor_stats(const at::GenericPackedTensorAccessor<float, 3, at::RestrictPtrTraits, int32_t>& accessor, const char* name) {
    float sum = 0.0f;
    float min_v = 1e10f;
    float max_v = -1e10f;
    int count = accessor.size(0) * accessor.size(1) * accessor.size(2);

    for (int i = 0; i < accessor.size(0); i++) {
        for (int j = 0; j < accessor.size(1); j++) {
            for (int k = 0; k < accessor.size(2); k++) {
                float val = accessor[i][j][k];
                sum += val;
                if (val < min_v) min_v = val;
                if (val > max_v) max_v = val;
            }
        }
    }

    float mean = sum / (count > 0 ? count : 1);
    printf("%s -> Size: [%d, %d, %d], Mean: %.4f, Min: %.4f, Max: %.4f\n",
           name, accessor.size(0), accessor.size(1), accessor.size(2), mean, min_v, max_v);
}


__device__ void print_accessor_stats(const at::GenericPackedTensorAccessor<float, 1, at::RestrictPtrTraits, int64_t>& accessor, const char* name) {
    float sum = 0.0f;
    float min_v = 1e10f;
    float max_v = -1e10f;
    int count = accessor.size(0);

    for (int i = 0; i < accessor.size(0); i++) {
        float val = accessor[i];
        sum += val;
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }

    float mean = sum / (count > 0 ? count : 1);
    printf("%s -> Size: [%d], Mean: %.4f, Min: %.4f, Max: %.4f\n",
           name, accessor.size(0), mean, min_v, max_v);
}

__device__ void print_accessor_stats(const at::GenericPackedTensorAccessor<float, 2, at::RestrictPtrTraits, int64_t>& accessor, const char* name) {
    float sum = 0.0f;
    float min_v = 1e10f;
    float max_v = -1e10f;
    int count = accessor.size(0) * accessor.size(1);

    for (int i = 0; i < accessor.size(0); i++) {
        for (int j = 0; j < accessor.size(1); j++) {
            float val = accessor[i][j];
            sum += val;
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
        }
    }

    float mean = sum / (count > 0 ? count : 1);
    printf("%s -> Size: [%d, %d], Mean: %.4f, Min: %.4f, Max: %.4f\n",
           name, accessor.size(0), accessor.size(1), mean, min_v, max_v);
}

__device__ void print_accessor_stats(const at::GenericPackedTensorAccessor<float, 3, at::RestrictPtrTraits, int64_t>& accessor, const char* name) {
    float sum = 0.0f;
    float min_v = 1e10f;
    float max_v = -1e10f;
    int count = accessor.size(0) * accessor.size(1) * accessor.size(2);

    for (int i = 0; i < accessor.size(0); i++) {
        for (int j = 0; j < accessor.size(1); j++) {
            for (int k = 0; k < accessor.size(2); k++) {
                float val = accessor[i][j][k];
                sum += val;
                if (val < min_v) min_v = val;
                if (val > max_v) max_v = val;
            }
        }
    }

    float mean = sum / (count > 0 ? count : 1);
    printf("%s -> Size: [%d, %d, %d], Mean: %.4f, Min: %.4f, Max: %.4f\n",
           name, accessor.size(0), accessor.size(1), accessor.size(2), mean, min_v, max_v);
}



///* apply a block reduction on a array of blockDim.x elements on a thread block (blockDim.x >= 64) */
//
//__device__ void block_reduce(float *smem, *float &partial_sum)
//{
//    /* data a loaded in shared memory before calling the function */
//
//    /*
//        Apply the pairwise method also called a tree reduction. The
//        number of active threads is halved between iteration.
//    */
//    for (offset = blockDim.x / 2; offset > 0; offset  /= 2) {
//        /* the number of active threads is halved after each iteration */
//        if (threadIdx.x < offset)
//            smem[threadIdx.x] += smem[threadIdx.x + offset];
//        /*  synchronization barrier */
//        __syncthreads();
//    }
//
//    if (threadIdx.x == 0)
//        partial_sum = smem[0];
//}
//
//
//__device__ int retirementCount = 0;
//
//__global__ void reduce_gpu_sptr(float *sdata, int size, float *res) {
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    extern __shared__ float smem[];
//
//    /* Split the data in smaller chunk. The block only only on one chunk */
//    if (tid >= size)
//        smem[threadIdx.x] = 0.0;
//    else
//        smem[threadIdx.x] = data[tid];
//
//    /* Wait for all thread to update shared memory */
//    __syncthreads();
//
//    /* apply the block reduction */
//    float partial_sum = 0.0;
//    block_reduce(smem, partial_sum);
//
//    if (threadIdx.x == 0)
//        partial_results[blockIdx.x] = partial_sum;
//
//    __threadfences();
//
//    bool __shared__ amLast = false;
//    if (threadIdx.x == 0) {
//        int prev = atomicInc(&retirementCount, gridSize.x);
//        amLast = (prev == (gridDim.x - 1));
//    }
//    __syncthreads();
//
//    if (amLast) {
//        if (threadIdx.x == 0) {
//            float sum = 0;
//            for (int i = 0; i < gridDim.x; i++)
//                sum += res[i];
//            res[0] = sum;
//        }
//    }
//}


__device__ void block_reduce(float *smem) {
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset)
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }
}

__global__ void reduce_first_dim(const float *data, int N, int vec_size, float *res) {
    extern __shared__ float smem[];

    int idx = blockIdx.x;  // index along final flattened
    int tid = threadIdx.x;

    if (idx >= vec_size)
        return;

    float local_sum = 0.0f;

    // Stride loop to handle large N
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += data[i * vec_size + idx];
    }
    smem[tid] = local_sum;
    __syncthreads();

    block_reduce(smem);

    if (tid == 0) {
        res[idx] = smem[0];
    }
}


void reduce_tensor(torch::Tensor& input_tensor) 
{
    // Make contiguous
    auto input_tensor_contig = input_tensor.contiguous();
    //1 thread for the number of elements in the first dimension
    int num_threads = input_tensor_contig.size(0);

    // Get shape of remaining dimensions after first
    auto shape = input_tensor_contig.sizes();
    int ndim = shape.size();

    // Compute vec_size (product of all dimensions except first)
    int vec_size = 1;
    for (int i = 1; i < ndim; ++i) {
        vec_size *= shape[i];
    }

    const float* input_ptr = input_tensor_contig.data_ptr<float>();

    auto options = torch::TensorOptions().dtype(input_tensor.dtype()).device(input_tensor.device());
    torch::Tensor output_flat = torch::empty({vec_size}, options);

    int threads_per_block = 256;
    int num_blocks = vec_size;


    // Launch CUDA kernel, with shared memory for block reduction
    reduce_first_dim<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
        input_ptr, num_threads, vec_size, output_flat.data_ptr<float>());

    // Build shape for final view (drop first dimension)
    std::vector<int64_t> out_shape;
    for (int i = 1; i < ndim; ++i) {
        out_shape.push_back(shape[i]);
    }

    // Reshape and assign back
    input_tensor = output_flat.view(out_shape);
}


// Deterministic sum along dim 0 on CPU
torch::Tensor reduce_cpu_fw(const torch::Tensor& input) {
    // Move to CPU if needed
    torch::Tensor input_cpu = input;
    if (input.device().is_cuda()) {
        input_cpu = input.cpu();
    }

    // Initialize output tensor with zeros, same shape as one slice
    torch::Tensor out = torch::zeros_like(input_cpu[0]);

    // Fixed-order loop: sum slices one by one
    const int64_t num = input_cpu.size(0);
    for (int64_t i = 0; i < num; ++i) {
        out += input_cpu[i];
    }

    // Move back to original device if needed
    if (input.device().is_cuda()) {
        out = out.to(input.device());
    }

    return out;
}

// Alternative deterministic CPU reduction (reverse order)
torch::Tensor reduce_cpu_bw(const torch::Tensor& input) {
    // Move to CPU if needed
    torch::Tensor input_cpu = input;
    if (input.device().is_cuda()) {
        input_cpu = input.cpu();
    }

    // Initialize output tensor with zeros, same shape as one slice
    torch::Tensor out = torch::zeros_like(input_cpu[0]);

    const int64_t num = input_cpu.size(0);
    // Reverse order loop: sum slices from last to first
    for (int64_t i = num - 1; i >= 0; --i) {
        out += input_cpu[i];
    }

    // Move back to original device if needed
    if (input.device().is_cuda()) {
        out = out.to(input.device());
    }

    return out;
}


#include <random>

torch::Tensor reduce_cpu_shuffle(const torch::Tensor& input) {
    // Move to CPU if needed
    torch::Tensor input_cpu = input;
    if (input.device().is_cuda()) {
        input_cpu = input.cpu();
    }

    // Initialize output tensor with zeros
    torch::Tensor out = torch::zeros_like(input_cpu[0]);

    const int64_t num = input_cpu.size(0);

    // Create shuffled index list
    std::vector<int64_t> indices(num);
    for (int64_t i = 0; i < num; ++i) {
        indices[i] = i;
    }

    // Shuffle using random device
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Accumulate in shuffled order
    for (int64_t idx : indices) {
        out += input_cpu[idx];
    }

    // Move back to original device if needed
    if (input.device().is_cuda()) {
        out = out.to(input.device());
    }

    return out;
}


template <class T> T kahanSummation(T *data, int size)
{
    T sum = data[0];
    T c   = (T)0.0;

    for (int i = 1; i < size; i++) {
        T y = data[i] - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }

    return sum;
}


__global__ void kahan_reduce_nd_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_threads,
    int inner_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inner_size) return;

    float sum = 0.0f;
    float c = 0.0f;

    for (int t = 0; t < num_threads; ++t) {
        int flat_idx = t * inner_size + idx;
        float y = input[flat_idx] - c;
        float temp = sum + y;
        c = (temp - sum) - y;
        sum = temp;
    }

    output[idx] = sum;
}


torch::Tensor kahan_reduce_dim0(torch::Tensor input) {
    TORCH_CHECK(input.dim() >= 1, "Input must be at least 1D");

    int num_threads = input.size(0);
    printf("Number of threads: %d\n", num_threads);
    auto output_shape = input.sizes().slice(1);  // remove dim 0
    printf("Output shape: ");
    for (const auto& dim : output_shape) {
        printf("%lld ", dim);
    }
    printf("\n");


    // flatten inner dims for generality
    int inner_size = 1;
    for (int i = 1; i < input.dim(); ++i)
        inner_size *= input.size(i);
    printf("Inner size: %d\n", inner_size);

    auto output = torch::zeros({inner_size}, input.options());
    printf("Output tensor initialized with shape: %s\n", output.sizes().vec().c_str());

    const int threads = 256;
    const int blocks = (inner_size + threads - 1) / threads;
    printf("Launching kernel with %d blocks and %d threads per block\n", blocks, threads);

    kahan_reduce_nd_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_threads,
        inner_size
    );

    // reshape to original shape minus dim 0
    return output.view(output_shape);
}

