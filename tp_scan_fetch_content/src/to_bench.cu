#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

template <typename T>
__global__
void kernel_scan_baseline(raft::device_span<T> buffer)
{
    for (int i = 1; i < buffer.size(); ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(rmm::device_uvector<int>& buffer)
{
	kernel_scan_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kogge_stone_scan(raft::device_span<T> buffer)
{
    // loading in shared might be faster idk kogge_stone is bad anyways
    // Not managing multiblock here
    int tid = threadIdx.x;
    for (int offset = 1; offset < buffer.size(); offset *= 2)
    {
        int to_add = 0;
        if (tid >= offset)
            to_add = buffer[tid - offset];
        __syncthreads();
        buffer[tid] += to_add;
        __syncthreads();
    }
}

template <typename T>
__global__
void brent_kung_scan(raft::device_span<T> buffer)
{
    // Reduce step
    int two_pow_i = 1;
    for (; two_pow_i < buffer.size(); two_pow_i *=2)
    {
        int j = two_pow_i * 2 * threadIdx.x - 1;
        if (j < buffer.size() && (j - two_pow_i) >=0)
            buffer[j] += buffer[j - two_pow_i];
        __syncthreads();
    }

    // Post-Reduce step
    // keep using offset to manage not power of two sizes
    for (int two_pow_i_plus_one = two_pow_i; two_pow_i_plus_one > 1; two_pow_i_plus_one /= 2)
    {
        int j = two_pow_i_plus_one + two_pow_i_plus_one * threadIdx.x;
        if (j + two_pow_i_plus_one/2 - 1 < buffer.size())
            buffer[j + two_pow_i_plus_one/2 - 1] += buffer[j - 1];
        __syncthreads();
    }
}

template <typename T>
__global__
void sklansky_scan(raft::device_span<T> buffer)
{
    extern __shared__ T sdata[];
    // Load data into shared memory

    sdata[threadIdx.x] = buffer[threadIdx.x];
    sdata[threadIdx.x + blockDim.x] = buffer[threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = 1; i < buffer.size(); i *= 2)
    {
        int j = i - 1 + 2 * (threadIdx.x - threadIdx.x % i);
        int k = threadIdx.x % i;
        sdata[j + k + 1] += sdata[j];
        __syncthreads();
    }

    // Write back to global memory
    buffer[threadIdx.x] = sdata[threadIdx.x];
    buffer[threadIdx.x + blockDim.x] = sdata[threadIdx.x + blockDim.x];

}

template <typename T>
__global__
void kernel_your_scan3(raft::device_span<T> buffer)
{
    // TODO
    // ...
}

void your_scan(rmm::device_uvector<int>& buffer)
{
    size_t size = buffer.size();
    size_t thread_per_block = std::min<size_t>(1024, size / 2); // Hardcoding max threads per block is REALLY faster
    size_t blocks_num = 1;
    size_t shared_memory_size = sizeof(int) * size;

	sklansky_scan<int><<<blocks_num, thread_per_block, shared_memory_size, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
    for (int i =0; i < buffer.size(); ++i)
    {
        int val;
        cudaMemcpy(&val, buffer.data() + i, sizeof(int), cudaMemcpyDeviceToHost);
        //printf("buffer[%d] = %d\n", i, val);
    }
}