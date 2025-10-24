#include "radix_sort.cuh"

#include <assert.h>
#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define NB_BINS 10


__global__ void radix_sort_kernel(raft::device_span<int> in, raft::device_span<int> out, int size, int iteration)
{
    int tid = threadIdx.x;
    // Get Blocks Local Histogram
    __shared__ int shared_hist[NB_BINS * 2 + 1]; // assuming NB_BINS elements
    int *inc_cum_hist = &shared_hist[NB_BINS + 1]; // inclusive cum hist
    int *exc_cum_hist = &shared_hist[NB_BINS]; // exclusive cum hist

    if (tid < NB_BINS * 2 + 1)
        shared_hist[tid] = 0;
    __syncthreads();

    if (tid < size)
    {
        int value = in[tid];
        for (int i = 0; i < iteration; ++i)
        {
            value /= NB_BINS;
        }
        atomicAdd(&shared_hist[value % NB_BINS], 1);
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

void radix_sort(std::vector<int>& array, int max_value)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int size = array.size();
    rmm::device_uvector<int> buff_1(size, stream);
    rmm::device_uvector<int> buff_2(size, stream);
    cudaMemcpyAsync(buff_1.data(), array.data(), size * sizeof(int), cudaMemcpyHostToDevice, stream);

    raft::device_span<int> d_span_1(buff_1.data(), buff_1.size());
    raft::device_span<int> d_span_2(buff_2.data(), buff_2.size());

    //size_t max_threads = 1024;
    //size_t thread_per_block = max_threads;
    //size_t threads = ;
    //size_t blocks = (threads + thread_per_block - 1) / thread_per_block;
    // each thread = 4 pixels and 4 predicate values
    //size_t shared_memory_size = thread_per_block * 4 * 2 * sizeof(int);
    for (int iteration = 0; max_value > 0; iteration++)
    {
        radix_sort_kernel<<<1, size, 0, stream>>>
            (d_span_1, d_span_2, size, iteration);
        max_value /= NB_BINS;
        std::swap(d_span_1, d_span_2);
    }

    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(array.data(), d_span_1.data(), array.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

bool test_radix_sort()
{
    int n = 1024; // nb of values in array
    std::vector<int> array(n);
    std::mt19937 rng(42);
    int max = 10000000;
    for (int& x : array)
        x = rng() % max;

    std::vector<int> base = array;
    std::vector<int> cpu_array = array;
    std::vector<int> gpu_array = array;

    // CPU sort timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(cpu_array.begin(), cpu_array.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU sort timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    radix_sort(gpu_array, max - 1);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    // Validation
    bool ok = std::is_sorted(gpu_array.begin(), gpu_array.end());
    bool matches = (cpu_array == gpu_array);

    std::cout << "nb values: " << n << "\n";
    std::cout << "max value: " << max << "\n";
    std::cout << "CPU sort time: " << cpu_time << " ms\n";
    std::cout << "GPU sort time: " << gpu_time << " ms\n";
    if (ok && matches)
        std::cout << "GPU sort is sorted ! ✅" << std::endl;
    else
    {
        std::cout << "GPU sort is NOT sorted ! ❌" << std::endl;
        for (int i = 0; i < n; ++i)
            std::cout << "CPU[" << i << "]=" << cpu_array[i] << " VS GPU[" << i << "]=" << gpu_array[i] << std::endl;
    }

    return ok;
}
