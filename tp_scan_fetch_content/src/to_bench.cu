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
void kernel_your_scan1(raft::device_span<T> buffer)
{

}

template <typename T>
__global__
void kernel_your_scan2(raft::device_span<T> buffer)
{
    // TODO
    // ...
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
    size_t thread_per_block = std::min<size_t>(1024, size); // Hardcoding max threads per block is REALLY faster
    size_t blocks_num = 1;
    size_t shared_memory_size = 0;

	kogge_stone_scan<int><<<blocks_num, thread_per_block, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}