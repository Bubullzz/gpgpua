#include "radix_sort.cuh"

#define NB_BINS 10
#define NB_VALUES_PER_BLOCK 4

__device__ int global_hist[NB_BINS];

__global__ void init_global_hist(raft::device_span<int> in, int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (tid < NB_BINS)
        global_hist[tid] = 0;
    __syncthreads();

    int pos = tid;
    int n = in.size();
    while (pos < n)
    {
        int value = in[pos];
        for (int i = 0; i < iteration; ++i)
        {
            value /= NB_BINS;
        }
        atomicAdd(&global_hist[value % NB_BINS], 1);
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
        int last = global_hist[0];
        global_hist[0] = 0;
        for (int i = 1; i < NB_BINS; ++i)
        {
            int tmp = global_hist[i];
            global_hist[i] = last + global_hist[i - 1];
            last = tmp;
        }
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < NB_BINS; ++i)
        {
            in[i] = global_hist[i];
        }
    }
    __syncthreads();
    return;
}

__global__ void radix_sort_kernel(raft::device_span<int> in,
    raft::device_span<int> out,
    int iteration,
    int* block_num)
{
    // Static shared memory allocation
    __shared__ int shared_hist[NB_BINS * 2 + 1]; // Room for local hist
    int *inc_cum_hist = &shared_hist[NB_BINS + 1]; // inclusive cum hist
    int *exc_cum_hist = &shared_hist[NB_BINS]; // exclusive cum hist
 
    cuda::atomic_ref<int, cuda::thread_scope_device> block(*block_num);

    if (threadIdx.x == 0)
    {
        my_block_id = block.fetch_add(1);
        to_add = 0;
    }
    __syncthreads();

    int block_base = blockIdx.x * NB_VALUES_PER_BLOCK;
    int tid = threadIdx.x;

    // Initialize shared memory
    int offset = tid;
    while(offset < NB_BINS * 2 + 1)
    {
        shared_hist[offset] = 0;
        offset += blockDim.x;
    }
    __syncthreads();

    // Build Local Histogram
    offset = tid;
    while (offset < NB_VALUES_PER_BLOCK)
    {
        int value = in[block_base + offset];
        for (int i = 0; i < iteration; ++i)
        {
            value /= NB_BINS;
        }
        atomicAdd(&shared_hist[value % NB_BINS], 1);
        offset += blockDim.x;
    }
    __syncthreads();


    // Load local histogram into inclusive cumulative histogram
    if (tid < NB_BINS)
    {
        inc_cum_hist[tid] = shared_hist[tid];
    }

    __syncthreads();


    // Compute local Exclusive Cumulative Histogram
    for (int i = 1; i < NB_BINS; i *= 2)
    {
        int j = i - 1 + 2 * (tid - tid % i);
        int k = tid % i;
        if (j + k + 1 < NB_BINS)
            inc_cum_hist[j + k + 1] += inc_cum_hist[j];
        __syncthreads();
    }
    __syncthreads();

    if (tid == 0)
        exc_cum_hist[0] = 0;
    
    __syncthreads();
    // Place the values in the correct position
    int my_threads_counter = 0;
    int my_threads_val = tid;
    for (int i = 0; i < size; ++i)
    {
        int base_value = in[i];
        int value = base_value;
        for (int it = 0; it < iteration; ++it)
        {
            value /= NB_BINS;
        }
        if (i < size && (value % NB_BINS) == my_threads_val)
        {
            out[exc_cum_hist[my_threads_val] + my_threads_counter] = base_value;
            my_threads_counter++;
        }
    }

    // And voila ? one round of radix sort done ?
    return;

        __syncthreads();
    if (tid ==0)
    {
        for (int i = 0; i < NB_BINS; ++i)
        {
            out[i] = exc_cum_hist[i];
        }
    }
    __syncthreads();
    return;



}

void radix_sort(rmm::device_uvector<int>& in, rmm::device_uvector<int>& out, int max_value, cudaStream_t stream)
{
    int size = in.size();

    size_t max_threads = 1024;
    size_t thread_per_block = max_threads;
    size_t nb_blocks = 64;
    rmm::device_scalar<int> block_num(0, stream);


    init_global_hist<<<nb_blocks, thread_per_block, 0, stream>>>(raft::device_span<int>(in.data(), size), 1);
    global_hist_cum_sum<<<1, 1, 0, stream>>>(raft::device_span<int>(in.data(), size));
    cudaMemcpyAsync(out.data(), in.data(), NB_BINS * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);


    size_t all_hists_size = NB_BINS * sizeof(int) * nb_blocks;
    rmm::device_uvector<int> states(nb_blocks, stream);
    rmm::device_uvector<int> local_hists(all_hists_size, stream);
    rmm::device_uvector<int> exclusive_prefix_summed_hists(all_hists_size, stream);
    return;

    //size_t threads = ;
    //size_t blocks = (threads + thread_per_block - 1) / thread_per_block;
    // each thread = 4 pixels and 4 predicate values
    //size_t shared_memory_size = thread_per_block * 4 * 2 * sizeof(int);
    for (int iteration = 0; max_value > 0; iteration++)
    {
        block_num.set_value(0);
        radix_sort_kernel<<<1, size, 0, stream>>>
            (raft::device_span<int>(in.data(), size),
            raft::device_span<int>(out.data(), size),
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
    int n = 20; // nb of values in array
    std::vector<int> array(n);
    std::mt19937 rng(42);
    int max = 10;
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

    rmm::device_uvector<int> my_gpu_d_in(cpu_array.size(), my_gpu_stream);
    rmm::device_uvector<int> my_gpu_d_out(cpu_array.size(), my_gpu_stream);
    cudaMemcpyAsync(my_gpu_d_in.data(), cpu_array.data(), cpu_array.size() * sizeof(int),
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
        if (true)
        {
            for (int i = 0; i < n; ++i)
            {
                if (cpu_array[i] != gpu_array[i] || cub_array[i] != gpu_array[i])
                {
                    std::cout << "Index " << i
                            << ": CPU=" << cpu_array[i]
                            << " GPU=" << gpu_array[i]
                            << " CUB=" << cub_array[i] << "\n";
                }
            }
        }
    }

    return ok && matches_cpu && matches_cub;
}