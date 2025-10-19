#include "fix_gpu_industrial.cuh"
#include "image.hh"

#include <assert.h>
#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

__device__
bool is_garbage(int val) {
    return val == -27;
}

void fix_image_gpu_industrial(Image& to_fix)
{
    return;
    cudaStream_t stream;
    cudaStreamCreate(&stream);    
    auto exec = thrust::cuda::par.on(stream);

    rmm::device_uvector<int> buffer1(to_fix.size(), stream);
    rmm::device_uvector<int> buffer2(to_fix.size(), stream);
    cudaMemcpy(buffer1.data(), to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice);


    // Remove -27 values 
    thrust::copy_if(
        exec,
        buffer1.begin(),
        buffer1.end(),
        buffer2.begin(),
        [] __device__ (int val) {
            return !is_garbage(val);
        }
    );

    // Apply add and substracts
    thrust::transform(
        exec,
        buffer2.begin(),
        buffer2.end(),
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

    cudaMemcpy(to_fix.buffer, buffer1.data(), to_fix.size() * sizeof(int), cudaMemcpyDeviceToHost);
}