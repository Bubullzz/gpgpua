#include "fix_gpu_industrial.cuh"
#include "image.hh"

#include <cub/cub.cuh>
#include <assert.h>
#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

struct hist_equalize_functor //: public thrust::unary_function<int, int>
{
    const int cdf_min;
    const float divider;
    const int* cum_hist;

    hist_equalize_functor(int _cdf_min, int _divider, const int* _cum_hist) : cdf_min(_cdf_min), divider(_divider), cum_hist(_cum_hist) {}

    __device__
    int operator()(int val) const {
        int cdf_pixel = cum_hist[val];
        return static_cast<int>((cdf_pixel - cdf_min) / divider * 255.0f + 0.5f); // cast<int>(x + 0.5f) <==> roundf(x)
    }
};

__device__
bool is_garbage(int val) {
    return val == -27;
}

// This function fixes the image and calculates the total pixel values
void fix_image_gpu_industrial(Image& to_fix)
{
    // Define the stream for current image
    cudaStream_t stream;
    cudaStreamCreate(&stream);    
    auto exec = thrust::cuda::par.on(stream);

    // Allocate device buffers
    rmm::device_uvector<int> buffer1(to_fix.size(), stream);
    rmm::device_uvector<int> buffer2(to_fix.size(), stream);
    rmm::device_uvector<int> cum_hist_buffer(256, stream);
    cudaMemcpy(buffer1.data(), to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice);


    // Remove -27 values 
    auto end = thrust::copy_if(
        exec,
        buffer1.begin(),
        buffer1.end(),
        buffer2.begin(),
        [] __device__ (int val) {
            return !is_garbage(val);
        }
    );
    size_t new_size = end - buffer2.begin();

    // Apply adds and substracts
    int4* buffer2_as_int4 = reinterpret_cast<int4*>(buffer2.begin());
    size_t num_int4 = (new_size + 3) / 4;
    int4* buffer1_as_int4 = reinterpret_cast<int4*>(buffer1.begin());
    thrust::transform(
        exec,
        buffer2_as_int4,
        buffer2_as_int4 + num_int4,
        buffer1_as_int4,
        [] __device__ (int4 val) {
            int4 res;
            res.x = val.x + 1;
            res.y = val.y - 5;
            res.z = val.z + 3;
            res.w = val.w - 8;
            return res;
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

    // Find cdf_min
    auto first_none_zero = thrust::find_if(
        exec,
        cum_hist_buffer.begin(),
        cum_hist_buffer.end(),
        [] __device__ (auto v) { return v != 0; });
    
    // Copy that value to host
    int cdf_min = 0;
    cudaMemcpy(&cdf_min, first_none_zero, sizeof(int), cudaMemcpyDeviceToHost);
    
    
    // Pre-compute all Hist-equalized values once. Could do that in CPU maybe, should test
    float divider = new_size - cdf_min;
    thrust::transform(
        exec,
        cum_hist_buffer.begin(),
        cum_hist_buffer.end(),
        cum_hist_buffer.begin(),
        [=] __device__ (int val) {
            return static_cast<int>((val - cdf_min) / divider * 255.0f + 0.5f); // cast<int>(x + 0.5f) <==> roundf(x)
        }
    );
    
    // Apply histogram equalization using the precomputed values
    thrust::gather(
        exec,
        buffer1.begin(),
        buffer1.begin() + new_size,
        cum_hist_buffer.begin(),
        buffer2.begin()
    );

    // Taking advantage of the fact that full image is already loaded in buffer2 to compute total
    int total = thrust::reduce(
        exec,
        buffer2.begin(),
        buffer2.begin() + new_size
    );

    to_fix.to_sort.total = total;

    // Will be useful when I refactor types to gain performances
    // to_fix.char_buffer = malloc(new_size * sizeof(char));

    // I Should dig into cudaMemcpyAsync 
    cudaMemcpy(to_fix.buffer, buffer2.data(), new_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Ok i guess i love thrust now
    return;
}
