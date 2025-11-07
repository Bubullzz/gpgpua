%%writefile radix_sort.cu
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// Can tweak these to try and find better perf
#define NB_BITS_MASK 10
#define NB_BINS (1 << NB_BITS_MASK)
#define MASK (NB_BINS - 1)
#define NB_VALUES_PER_BLOCK 2048
#define THREAD_PER_BLOCK 1024

__device__ int global_hist[NB_BINS];

// Just sets all of the histograms values to 0
__global__ void set_global_hist_to_0()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int pos = tid;
    while (pos < NB_BINS)
    {
        global_hist[pos] = 0;
        pos += stride;
    }
}


// Returns the right bin at the given iteration for a certain value
__device__ int get_bin_place(int value, int iteration)
{
    return (value >> (iteration * NB_BITS_MASK)) & MASK;
}


__global__ void init_global_hist(const int* in, int n, int iteration)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (pos < n)
    {
        int value = get_bin_place(in[pos], iteration);
        atomicAdd(&global_hist[value], 1);
        pos += stride;
    }

    __syncthreads();
    return;

}

// slow version bc i wanna focus on radix sort first
__global__ void global_hist_cum_sum()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int prev = 0;
        for (int i = 0; i < NB_BINS; ++i)
        {
            int current = global_hist[i];
            global_hist[i] = prev;
            prev += current;
        }
    }
    __syncthreads();
    return;
}

__global__ void radix_sort_kernel(
    const int* in,
    int* out,
    int* states,
    int* local_hists,
    int* prefix_summed_hists,
    int n, int iteration, int* block_num)
{
    // Static shared memory allocation
    __shared__ int smem[NB_BINS * 2]; // Room for local hist and prefix sum that gets computed, pretty sure i can divide this by two with some shenanigans
    int *my_local_hist = &smem[0];
    int *my_exclusive_prefix_sum = &smem[NB_BINS];
 
    __shared__ int my_block_id;
    __shared__ int current_flag;
    
    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> block(*block_num);
        my_block_id = block.fetch_add(1);
    }
    __syncthreads();

    int block_base = my_block_id * NB_VALUES_PER_BLOCK;
    int tid = threadIdx.x;

    // Initialize shared memory
    int offset = tid;
    while(offset < NB_BINS * 2)
    {
        smem[offset] = 0;
        offset += blockDim.x;
    }
    __syncthreads();

    // Build Local Histogram
    offset = tid;
    while (offset < NB_VALUES_PER_BLOCK)
    {
        int value = get_bin_place(in[block_base + offset], iteration);
        atomicAdd(&my_local_hist[value], 1);
        offset += blockDim.x;
    }
    __syncthreads();

    // Make my local histogram available
    offset = tid;
    while (offset < NB_BINS)
    {
        local_hists[my_block_id * NB_BINS + offset] = my_local_hist[offset];
        offset += blockDim.x;
    }
    __syncthreads();

    // Inform other blocks that my local histogram is ready
    if (tid == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> my_state(states[my_block_id]);
        my_state.store(1, cuda::memory_order_release); // computing state
        my_state.notify_all();
    }

    int currently_looking_at = my_block_id - 1;
    // Compute prefix sum of previous histograms
    while (true)
    {
        if (currently_looking_at < 0)
            break; // We have got the total prefix sum
        if (threadIdx.x == 0)
        {
            cuda::atomic_ref<int, cuda::thread_scope_device> state(states[currently_looking_at]);
            while (state.load(cuda::memory_order_acquire) == 0)
                state.wait(0);
            current_flag = state.load(cuda::memory_order_acquire);
        }
        __syncthreads(); // All threads wait for thread_0 to give the go

        int* current_hist;
        if (current_flag == 1) // local value ready
            current_hist = &local_hists[NB_BINS * currently_looking_at];
        else // global value ready
            current_hist = &prefix_summed_hists[NB_BINS * currently_looking_at];

        // Add the new histogram to current sum
        offset = tid;
        while (offset < NB_BINS)
        {
            int val_to_add = current_hist[offset];
            my_exclusive_prefix_sum[offset] += val_to_add;
            offset += blockDim.x;
        }
        __syncthreads();
        if (current_flag == 2)
            break; // Get out !!! We just computed the total prefix sum !

        currently_looking_at--;
    }
    __syncthreads();

    // So now, we have shared memory filled with our local histogram then the exclusive prefix sum

    // Let's make our prefix sum available
    offset = tid;
    while (offset < NB_BINS)
    {
        prefix_summed_hists[my_block_id * NB_BINS + offset] = my_exclusive_prefix_sum[offset] + my_local_hist[offset];
        offset += blockDim.x;
    }    
    __syncthreads();

    // Now prefix_summed_hists[my_block_id * NB_BINS] contains the full inclusive prefix sum from block_0 to my_block_id
    // So our state is 2 : done !
    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> my_state(states[my_block_id]);
        my_state.store(2);
        my_state.notify_all();
    }

    // We have global offset from global_hist, local offset in my_exclusive_prefix_sum, you know what it means... #heheha
    int base_offset = NB_VALUES_PER_BLOCK * my_block_id;
    const int NB_VALUES_PER_THREAD = (NB_BINS + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    int already_seen[NB_VALUES_PER_THREAD];
    for (int k = 0; k < NB_VALUES_PER_THREAD; ++k) already_seen[k] = 0;

    // All the threads read the value and the right thread puts it at the right place
    for (int i = 0; i < NB_VALUES_PER_BLOCK; ++i)
    {
        if (base_offset + i >= n)
            continue;
        int base_value = in[base_offset + i];
        int value = get_bin_place(base_value, iteration);  
        if (value % THREAD_PER_BLOCK == threadIdx.x)
        {
            int destination = global_hist[value] + my_exclusive_prefix_sum[value] + already_seen[value / THREAD_PER_BLOCK];
            already_seen[value / THREAD_PER_BLOCK] += 1;
            out[destination] = base_value;
        }
    }

    // And voila ? one round of radix sort done ?
    return;
}

void radix_sort(int* d_in_init, int* d_out_init, int n, int max_value, cudaStream_t stream)
{
    size_t thread_per_block = THREAD_PER_BLOCK;
    size_t nb_blocks = (n + NB_VALUES_PER_BLOCK - 1) / NB_VALUES_PER_BLOCK;

    size_t all_hists_size = NB_BINS * nb_blocks;
    int* d_block_num = nullptr;
    int* d_states = nullptr;
    int* d_local_hists = nullptr;
    int* d_prefix_summed_hists = nullptr;
    
    cudaMalloc(&d_block_num, sizeof(int));
    cudaMalloc(&d_states, nb_blocks * sizeof(int));
    cudaMalloc(&d_local_hists, all_hists_size * sizeof(int));
    cudaMalloc(&d_prefix_summed_hists, all_hists_size * sizeof(int));

    std::cout << "blocks number :" << nb_blocks << std::endl;
    
    int* d_in = d_in_init;
    int* d_out = d_out_init;

    for (int iteration = 0; max_value > 0; iteration++)
    {
        // Reinitialize memory
        cudaMemsetAsync(d_block_num, 0, sizeof(int), stream);
        cudaMemsetAsync(d_states, 0, nb_blocks * sizeof(int), stream);
        cudaMemsetAsync(d_local_hists, 0, all_hists_size * sizeof(int), stream);
        cudaMemsetAsync(d_prefix_summed_hists, 0, all_hists_size * sizeof(int), stream);

        // Re-compute global histogram very slow for the moment but just need working code for that
        // This is around 10 ms rn 
        set_global_hist_to_0<<<64, thread_per_block, 0, stream>>>(); // Arbitrary launch with 64 blocks could tweak it later
        
        init_global_hist<<<1, thread_per_block, 0, stream>>>(d_in, n, iteration);
        global_hist_cum_sum<<<1, 1, 0, stream>>>();

        // Compute radix sort !
        radix_sort_kernel<<<nb_blocks, thread_per_block, 0, stream>>>(
            d_in,
            d_out, 
            d_states, 
            d_local_hists, 
            d_prefix_summed_hists,
            n, 
            iteration, 
            d_block_num);
        max_value /= NB_BINS;
        cudaStreamSynchronize(stream);
        int* tmp = d_in; d_in = d_out; d_out = tmp;

    }
    int* tmp = d_in; d_in = d_out; d_out = tmp;

    cudaFree(d_block_num);
    cudaFree(d_states);
    cudaFree(d_local_hists);
    cudaFree(d_prefix_summed_hists);
}

// Just run this function to test perfs and correctness
int main()
{
    int n = 2000000; // nb of values in array
    std::vector<int> array(n);
    std::mt19937 rng(42);
    int max = 100000;
    for (int& x : array)
        x = rng() % max;

    std::vector<int> cpu_array = array;
    std::vector<int> gpu_array = array;

    // ---------------------------
    // CPU sort timing
    // ---------------------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(cpu_array.begin(), cpu_array.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // ---------------------------
    // My GPU radix sort
    // ---------------------------
    cudaStream_t my_gpu_stream;
    cudaStreamCreate(&my_gpu_stream);

    int* d_in = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMemcpyAsync(d_in, gpu_array.data(), n * sizeof(int), cudaMemcpyHostToDevice, my_gpu_stream);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    radix_sort(d_in, d_out, n, max - 1, my_gpu_stream);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    cudaMemcpyAsync(gpu_array.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost, my_gpu_stream);
    cudaStreamSynchronize(my_gpu_stream);
    cudaStreamDestroy(my_gpu_stream);

    cudaFree(d_in);
    cudaFree(d_out);
    // ---------------------------
    // Validation
    // ---------------------------
    bool ok = std::is_sorted(gpu_array.begin(), gpu_array.end());
    bool matches_cpu = (cpu_array == gpu_array);

    std::cout << "nb values: " << n << "\n";
    std::cout << "max value: " << max << "\n";
    std::cout << "CPU sort time: " << cpu_time << " ms\n";
    std::cout << "My  sort time: " << gpu_time << " ms\n";

    if (ok && matches_cpu)
        std::cout << "✅ GPU radix sort matches CPU results!" << std::endl;
    else
    {
        std::cout << "❌ GPU sort mismatch!\n";
        if (false)
        {
            for (int i = 0; i < n; ++i)
            {
                    std::cout << "Index " << i
                            << ": CPU=" << cpu_array[i]
                            << " Base=" << array[i]
                            << " GPU=" << gpu_array[i] << "\n";
            }
        }
    }

    return ok && matches_cpu;
}