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

#define NB_BINS = 10


__global__ void radix_sort_kernel(raft::device_span<int> array, int size)
{

}

void radix_sort(std::vector<int>& array)
{

}

bool test_radix_sort()
{
    int n = 20; // nb of values in array
    std::vector<int> array(n);
    std::mt19937 rng(42);
    int max = 1000;
    for (int& x : array)
        x = rng() % max;

    std::vector<int> cpu_array = array;
    std::vector<int> gpu_array = array;

    // CPU sort timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(cpu_array.begin(), cpu_array.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU sort timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    radix_sort(gpu_array);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    // Validation
    bool ok = std::is_sorted(gpu_array.begin(), gpu_array.end());

    std::cout << "nb values: " << n << "\n";
    std::cout << "CPU sort time: " << cpu_time << " ms\n";
    std::cout << "GPU sort time: " << gpu_time << " ms\n";
    if (ok)
        std::cout << "GPU sort is sorted ! ✅" << std::endl;
    else
    {
        std::cout << "GPU sort is NOT sorted ! ❌" << std::endl;
        for (int i = 0; i < n; ++i)
            std::cout << "CPU[" << i << "]=" << cpu_array[i] << " VS GPU[" << i << "]=" << gpu_array[i] << std::endl;
    }

    return ok;
}
