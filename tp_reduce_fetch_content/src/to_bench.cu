#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>


template <typename T>
__global__
void kernel_reduce_baseline(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    for (int i = 0; i < buffer.size(); ++i)
        *total.data() += buffer[i];
}

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total)
{
	kernel_reduce_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

__device__ void warp_unroll(int* shared, int pos){
    shared[pos] += shared[pos + 32];__syncwarp();
    shared[pos] += shared[pos + 16];__syncwarp();
    shared[pos] += shared[pos + 8]; __syncwarp();
    shared[pos] += shared[pos + 4]; __syncwarp();
    shared[pos] += shared[pos + 2]; __syncwarp();
    shared[pos] += shared[pos + 1]; __syncwarp();
}

template <typename T>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    extern __shared__ T sdata[];

    size_t buf_size = buffer.size();
    size_t pos = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    size_t stride = blockDim.x * 2 * gridDim.x;

    // Strided loading (didn't give me much speedup maybe something wrong ? Maybe we need bigger problem for it to matter ?)
    sdata[threadIdx.x] = 0;
    while (pos < buf_size){
        sdata[threadIdx.x] += buffer[pos] + buffer[pos + blockDim.x];
        pos += stride;
    }

    __syncthreads();
    #pragma unroll
    for (int i = blockDim.x/2; i > 32; i/=2)
    {
        if (threadIdx.x < i){
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        else
            break; // will take full warps out so no divergence and skips a few operations for them
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        warp_unroll(sdata, threadIdx.x);
    }

    // Using atomic add to avoid multiple kernel launches and is actually faster than 2 kernel launches lol
    if (threadIdx.x == 0)
        atomicAdd(total.data(), sdata[0]);
}

void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    size_t size = buffer.size() / 2; 
    size_t thread_per_block = std::min<size_t>(1024, size); // Hardcoding max threads per block is REALLY faster than querying the gpu
    size_t blocks_num = 8; // enter wwhatever size, will increase work per thread if lower, i think power of 2 should be better
    size_t shared_memory_size = sizeof(int) * thread_per_block;

    kernel_your_reduce<int><<<blocks_num, thread_per_block, shared_memory_size, buffer.stream()>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
    }