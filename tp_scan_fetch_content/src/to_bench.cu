#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>
#include <cuda/atomic>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

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

// NOT OPTIMIZED AND NO LOOKBACK
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

// NOT OPTIMIZED AND NO LOOKBACK
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
    // Sorry for readability !!!!
    for (int two_pow_i_plus_one = two_pow_i; two_pow_i_plus_one > 1; two_pow_i_plus_one /= 2)
    {
        int j = two_pow_i_plus_one + two_pow_i_plus_one * threadIdx.x;
        if (j + two_pow_i_plus_one/2 - 1 < buffer.size())
            buffer[j + two_pow_i_plus_one/2 - 1] += buffer[j - 1];
        __syncthreads();
    }
}

// A bit optimized and looking back :3
// Assuming right padding to not have to do bounds checking
template <typename T>
__global__
void sklansky_scan(raft::device_span<T> buffer,
                   int* block_num,
                   raft::device_span<int> states,
                   raft::device_span<T> local_sums,
                   raft::device_span<T> global_sums)
{
    extern __shared__ T sdata[];
    __shared__ int my_block_id;
    __shared__ int to_add;

    cuda::atomic_ref<int, cuda::thread_scope_device> block(*block_num);

    if (threadIdx.x == 0)
    {
        my_block_id = block.fetch_add(1);
        to_add = 0;
    }
    __syncthreads();
    // Load data into shared memory

    sdata[threadIdx.x] = buffer[my_block_id * blockDim.x * 2 + threadIdx.x];
    sdata[threadIdx.x + blockDim.x] = buffer[my_block_id * blockDim.x * 2 + threadIdx.x + blockDim.x];
    __syncthreads();

    for (int i = 1; i < blockDim.x * 2; i *= 2)
    {
        // Thanks to the paper and a headache here is the indexing
        int j = i - 1 + 2 * (threadIdx.x - threadIdx.x % i);
        int k = threadIdx.x % i;
        sdata[j + k + 1] += sdata[j];
        __syncthreads();
    }

    // Block result is done, starting decoupled lookback
    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> my_state(states[my_block_id]);
        cuda::atomic_ref<int, cuda::thread_scope_device> my_local_value(local_sums[my_block_id]);
        cuda::atomic_ref<int, cuda::thread_scope_device> my_global_value(global_sums[my_block_id]);
        my_local_value.store(sdata[blockDim.x * 2 - 1], cuda::memory_order_release); // last element of the block
        my_state.store(1, cuda::memory_order_release); // computing state
        my_state.notify_all();

        for (int b = my_block_id - 1; b >= 0; --b)
        {
            cuda::atomic_ref<int, cuda::thread_scope_device> state(states[b]);

            while (state.load(cuda::memory_order_acquire) == 0)
                state.wait(0);
            // state is either 1 (waiting) or 2 (done)

            if (state.load(cuda::memory_order_acquire) == 1) // local value ready
            {
                to_add += local_sums[b];
            }
            else // global value ready
            {
                to_add += global_sums[b];
                break;
            }
        }
        my_global_value.store(sdata[blockDim.x * 2 - 1] + to_add, cuda::memory_order_release); // last element of the block
        my_state.store(2, cuda::memory_order_release); // done state
        my_state.notify_all();
    }

    __syncthreads();
    buffer[my_block_id * blockDim.x * 2 + threadIdx.x] = sdata[threadIdx.x] + to_add;
    buffer[my_block_id * blockDim.x * 2 + threadIdx.x + blockDim.x] = sdata[threadIdx.x + blockDim.x] + to_add;
}

void your_scan(rmm::device_uvector<int>& buffer)
{
    size_t size = buffer.size() / 2; // assuming even size for simplicity and computing 2 elements per thread
    size_t threads_per_block = std::min<size_t>(1024, size); // 1024 threads bc it's the max on these machines
    size_t nb_blocks = (size + threads_per_block -1) / (threads_per_block);
    size_t shared_memory_size = sizeof(int) * threads_per_block * 2;

    rmm::device_scalar<int> block_num(0, buffer.stream());
    rmm::device_uvector<int> states(nb_blocks, buffer.stream()); // Could use uint8_t but does not work with atomics
    rmm::device_uvector<int> local_sums(nb_blocks, buffer.stream());
    rmm::device_uvector<int> global_sums(nb_blocks, buffer.stream());

    // Init all states to 0 at once
    cudaMemsetAsync(states.data(), 0, nb_blocks * sizeof(int), buffer.stream());

    sklansky_scan<int><<<nb_blocks, threads_per_block, shared_memory_size, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        block_num.data(),
        raft::device_span<int>(states.data(), nb_blocks),
        raft::device_span<int>(local_sums.data(), nb_blocks),
        raft::device_span<int>(global_sums.data(), nb_blocks));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}