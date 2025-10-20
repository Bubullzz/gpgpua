#include "fix_gpu_industrial.cuh"
#include "image.hh"

#include <cub/cub.cuh>
#include <assert.h>
#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

struct hist_equalize_functor //: public thrust::unary_function<int, int>
{
    const int cdf_min;
    const float divider;
    const int* cum_hist;

    hist_equalize_functor(int _cdf_min, int _divider, const int* _cum_hist) : cdf_min(_cdf_min), divider(_divider), cum_hist(_cum_hist) {}

    __host__ __device__
    int operator()(int val) const {
        int cdf_pixel = cum_hist[val];
        return static_cast<int>((cdf_pixel - cdf_min) / divider * 255.0f + 0.5f); // cast + 0.5f == roundf
    }
};

__device__
bool is_garbage(int val) {
    return val == -27;
}

void fix_image_gpu_industrial(Image& to_fix)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);    
    auto exec = thrust::cuda::par.on(stream);

    rmm::device_uvector<int> buffer1(to_fix.size(), stream);
    rmm::device_uvector<int> buffer2(to_fix.size(), stream);
    rmm::device_uvector<int> cum_hist_buffer(256, stream);
    cudaMemcpy(buffer1.data(), to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice);


    // Remove -27 values 
    auto end =thrust::copy_if(
        exec,
        buffer1.begin(),
        buffer1.end(),
        buffer2.begin(),
        [] __device__ (int val) {
            return !is_garbage(val);
        }
    );
    size_t new_size = end - buffer2.begin();
    // Apply add and substracts
    thrust::transform(
        exec,
        buffer2.begin(),
        buffer2.begin() + new_size,
        thrust::counting_iterator<int>(0), // indices
        buffer1.begin(),
        [] __device__ (int val, int idx) {
            switch (idx % 4) {
                case 0: return val + 1;
                case 1: return val - 5;
                case 2: return val + 3;
                case 3: return val - 8;
            }
            return val; // should not happen
        }
    );
    
    // Get histogram 
    void* tmp_storage_bytes = nullptr;
    size_t tmp_storage_bytes_num = 0;
    
    auto d_samples = buffer1.data();
    auto d_histogram = buffer2.data();
    auto num_levels = 257; // 256 bins + 1 WHATEVER I DONT KNOW I DONT WANT TO KNOW
    auto lower_level = 0;
    auto upper_level = 256;
    auto num_samples = new_size;
    
    // Apparently this first call only gives us the tmp_storage_bytes_num
    cub::DeviceHistogram::HistogramEven(
        tmp_storage_bytes, tmp_storage_bytes_num,
        d_samples, d_histogram,
        num_levels, lower_level, upper_level, num_samples);
    cudaMalloc(&tmp_storage_bytes, tmp_storage_bytes_num);
        
    // And now it computes the histogram for real
    cub::DeviceHistogram::HistogramEven(
        tmp_storage_bytes, tmp_storage_bytes_num,
        d_samples, d_histogram,
        num_levels, lower_level, upper_level, num_samples);
        
    cudaFree(tmp_storage_bytes);

    // Get Cumulative Histogram
    thrust::inclusive_scan(
        exec,
        buffer2.begin(),
        buffer2.begin() + 256,
        cum_hist_buffer.begin()
    );

    auto first_none_zero = thrust::find_if(
        exec,
        cum_hist_buffer.begin(),
        cum_hist_buffer.end(),
        [] __device__ (auto v) { return v != 0; });
    
    // Copy that value to host
    int cdf_min = 0;
    cudaMemcpy(&cdf_min, first_none_zero, sizeof(int), cudaMemcpyDeviceToHost);

    float divider = static_cast<float>(new_size - cdf_min);
    
    // Copy last value of cumulative histogram to host for debugging
    int cum_last = 0;
    cudaMemcpy(&cum_last, cum_hist_buffer.data() + 255, sizeof(int), cudaMemcpyDeviceToHost);

    thrust::transform(
        exec,
        buffer1.begin(),
        buffer1.begin() + new_size,
        buffer2.begin(),
        hist_equalize_functor(cdf_min, divider, cum_hist_buffer.data())
    );

    cudaMemcpy(to_fix.buffer, buffer2.data(), new_size * sizeof(int), cudaMemcpyDeviceToHost);
    return;
}
