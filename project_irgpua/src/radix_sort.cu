#include "radix_sort.cuh"
#include <cuda/atomic>

#define NB_BINS 2048
#define NB_VALUES_PER_BLOCK 1024
#define THREAD_PER_BLOCK 1024

__device__ int global_hist[NB_BINS];

__global__ void print_global_hist()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("global_hist:\n");
        for (int i = 0; i < NB_BINS; ++i)
        {
            printf("[%d] = %d\n", i, global_hist[i]);
        }
    }
}

__device__ int get_bin_place(int value, int iteration)
{
    for (int i = 0; i < iteration; ++i)
    {
        value /= NB_BINS;
    }
    return value % NB_BINS;
}

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

__global__ void init_global_hist(raft::device_span<int> in, int iteration)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int n = in.size();
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

// dummy version for the moment
__global__ void global_hist_cum_sum(raft::device_span<int> in)
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

__global__ void radix_sort_kernel(raft::device_span<int> in,
    raft::device_span<int> out,
    raft::device_span<int> states,
    raft::device_span<int> local_hists,
    raft::device_span<int> prefix_summed_hists,
    int iteration,
    int* block_num)
{
    // Static shared memory allocation
    __shared__ int smem[NB_BINS * 2]; // Room for local hist and prefix sum that gets computed
    int *my_local_hist = &smem[0];
    int *my_exclusive_prefix_sum = &smem[NB_BINS]; // Room for local hist and prefix sum that gets computed

    //int *inc_cum_hist = &shared_hist[NB_BINS + 1]; // inclusive cum hist
    //int *exc_cum_hist = &shared_hist[NB_BINS]; // exclusive cum hist
 
    __shared__ int my_block_id;
    __shared__ int current_flag;
    cuda::atomic_ref<int, cuda::thread_scope_device> block(*block_num);

    if (threadIdx.x == 0)
    {
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

    // Now prefix_summed_hists[my_block_id * NB_BINS] contains the full inclusive prefix sum from block_0 to my_block_id
    // So our state is 2 : done !
    __syncthreads();
    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> my_state(states[my_block_id]);
        my_state.store(2);
        my_state.notify_all();
    }
    __syncthreads();

    // We have global offset from global_hist, local offset in my_exclusive_prefix_sum, you know what it means... #heheha
    int base_offset = NB_VALUES_PER_BLOCK * my_block_id;
    const int nb_values_per_thread = (NB_BINS + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    int already_seen[nb_values_per_thread];
    for (int k = 0; k < nb_values_per_thread; ++k) already_seen[k] = 0;

    // All the threads read the value and the right thread puts it at the right place
    for (int i = 0; i < NB_VALUES_PER_BLOCK; ++i)
    {
        if (base_offset + i >= in.size())
            return;
        int base_value = in[base_offset + i];
        int value = get_bin_place(base_value, iteration);  
        if (value % THREAD_PER_BLOCK == threadIdx.x)
        {
            int destination = global_hist[value] + my_exclusive_prefix_sum[value] + already_seen[value / THREAD_PER_BLOCK];
            //printf("i = %d, writing %d to %d with glob_hist_val = %d, me_ex_pref = %d, already_seen = %d\n", i, base_value, destination, 
            //    global_hist[value], my_exclusive_prefix_sum[value], already_seen[value / THREAD_PER_BLOCK]);
            already_seen[value / THREAD_PER_BLOCK] += 1;
            out[destination] = base_value;
        }
        __syncthreads();

    }



    // And voila ? one round of radix sort done ?
    return;
}

void radix_sort(rmm::device_uvector<int>& in, rmm::device_uvector<int>& out, int max_value, cudaStream_t stream)
{
    int size = in.size();

    size_t thread_per_block = THREAD_PER_BLOCK;
    size_t nb_blocks = (size + NB_VALUES_PER_BLOCK - 1) / NB_VALUES_PER_BLOCK;

    size_t all_hists_size = NB_BINS * nb_blocks;
    rmm::device_scalar<int> block_num(0, stream);
    rmm::device_uvector<int> states(nb_blocks, stream);
    rmm::device_uvector<int> local_hists(all_hists_size, stream);
    rmm::device_uvector<int> prefix_summed_hists(all_hists_size, stream);
    
    std::cout << "blocks number :" << nb_blocks << std::endl;

    for (int iteration = 0; max_value > 0; iteration++)
    {
        // Reinitialize memory
        cudaMemsetAsync(block_num.data(), 0, sizeof(int));
        cudaMemsetAsync(states.data(), 0, states.size() * sizeof(int), stream);
        cudaMemsetAsync(local_hists.data(), 0, local_hists.size() * sizeof(int), stream);
        cudaMemsetAsync(prefix_summed_hists.data(), 0, prefix_summed_hists.size() * sizeof(int), stream);
        
        // Re-compute global histogram very slow for the moment but just need working code for that
        // This is around 10 ms rn 
        set_global_hist_to_0<<<64, thread_per_block, 0, stream>>>(); // Arbitrary launch with 64 blocks could tweak it later
        init_global_hist<<<1, thread_per_block, 0, stream>>>(raft::device_span<int>(in.data(), size), iteration);
        global_hist_cum_sum<<<1, 1, 0, stream>>>(raft::device_span<int>(in.data(), size));

        // Compute radix sort !
        radix_sort_kernel<<<nb_blocks, thread_per_block, 0, stream>>>(
            raft::device_span<int>(in.data(), size),
            raft::device_span<int>(out.data(), size),
            raft::device_span<int>(states.data(), size),
            raft::device_span<int>(local_hists.data(), size),
            raft::device_span<int>(prefix_summed_hists.data(), size),
            iteration, block_num.data());
        max_value /= NB_BINS;
        cudaStreamSynchronize(stream);
        std::swap(in, out);
    }
    //size_t threads = ;
    //size_t blocks = (threads + thread_per_block - 1) / thread_per_block;
    // each thread = 4 pixels and 4 predicate values
    //size_t shared_memory_size = thread_per_block * 4 * 2 * sizeof(int);
    for (int iteration = 0; max_value > 0; iteration++)
    {
        // Make sure to reset values, memories, flags etc... !!!!!!!!!!!!!!
        //block_num.set_value(0); // this does not work go cuda memset je pense
        radix_sort_kernel<<<1, size, 0, stream>>>
            (raft::device_span<int>(in.data(), size),
            raft::device_span<int>(out.data(), size),
            raft::device_span<int>(states.data(), size),
            raft::device_span<int>(local_hists.data(), size),
            raft::device_span<int>(prefix_summed_hists.data(), size),
            iteration, block_num.data());
        
            max_value /= NB_BINS;
        cudaStreamSynchronize(stream);
        std::swap(in, out);
    }

    std::swap(in, out);
}

// Just run this function to test perfs and correctness
bool test_radix_sort()
{
    int n = 2000000; // nb of values in array
    std::vector<int> array(n);
    std::mt19937 rng(42);
    int max = 100000;
    for (int& x : array)
        x = rng() % max;

    std::vector<int> cpu_array = array;
    std::vector<int> gpu_array = array;
    std::vector<int> cub_array = array;

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

    rmm::device_uvector<int> my_gpu_d_in(gpu_array.size(), my_gpu_stream);
    rmm::device_uvector<int> my_gpu_d_out(gpu_array.size(), my_gpu_stream);
    cudaMemcpyAsync(my_gpu_d_in.data(), gpu_array.data(), gpu_array.size() * sizeof(int),
                    cudaMemcpyHostToDevice, my_gpu_stream);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    radix_sort(my_gpu_d_in, my_gpu_d_out, max - 1, my_gpu_stream);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    cudaMemcpyAsync(gpu_array.data(), my_gpu_d_out.data(), gpu_array.size() * sizeof(int),
                    cudaMemcpyDeviceToHost, my_gpu_stream);
    cudaStreamDestroy(my_gpu_stream);

    // ---------------------------
    // CUB reference GPU sort
    // ---------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    rmm::device_uvector<int> d_in(cub_array.size(), stream);
    rmm::device_uvector<int> d_out(cub_array.size(), stream);
    cudaMemcpyAsync(d_in.data(), cub_array.data(), cub_array.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Query temp storage size
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_in.data(), d_out.data(), n, 0, sizeof(int) * 8, stream);

    // Allocate temp storage
    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

    auto start_cub = std::chrono::high_resolution_clock::now();
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_in.data(), d_out.data(), n, 0, sizeof(int) * 8, stream);
    cudaStreamSynchronize(stream);
    auto end_cub = std::chrono::high_resolution_clock::now();

    double cub_time = std::chrono::duration<double, std::milli>(end_cub - start_cub).count();

    cudaMemcpyAsync(cub_array.data(), d_out.data(), cub_array.size() * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_temp_storage, stream);
    cudaStreamDestroy(stream);

    // ---------------------------
    // Validation
    // ---------------------------
    bool ok = std::is_sorted(gpu_array.begin(), gpu_array.end());
    bool matches_cpu = (cpu_array == gpu_array);
    bool matches_cub = (cub_array == gpu_array);

    std::cout << "nb values: " << n << "\n";
    std::cout << "max value: " << max << "\n";
    std::cout << "CPU sort time: " << cpu_time << " ms\n";
    std::cout << "My  sort time: " << gpu_time << " ms\n";
    std::cout << "CUB sort time: " << cub_time << " ms\n";

    if (ok && matches_cpu && matches_cub)
        std::cout << "✅ GPU radix sort matches both CPU & CUB results!" << std::endl;
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

    return ok && matches_cpu && matches_cub;
}