#include "fix_gpu_handmade.cuh"
#include "image.hh"

#include <assert.h>
#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

// Kernel should be launched with 4 values per thread so the mask can be applied without a modulo operation
__global__ void fix_image(raft::device_span<int> buffer) {
    extern __shared__ int shared[];
    int* predicate = &shared[blockDim.x * 4];
    int idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    int size = buffer.size();

    // Load data into shared memory (should coalesce it later)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        shared[threadIdx.x * 4 + i] = buffer[idx + i];
    }
    __syncthreads();

    // Get Predicate
    constexpr int garbage_val = -27;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (shared[threadIdx.x * 4 + i] != garbage_val) {
            predicate[threadIdx.x * 4 + i] = 1;
        } else {
            predicate[threadIdx.x * 4 + i] = 0;
        }
    }
    __syncthreads();
    
    // bad scan for tests
    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x * 4; ++i) {
            predicate[i] += predicate[i - 1];
        }

    }
    __syncthreads();

    // NEED TO PROPAGATE SCAN FOR EVERY BLOCK

    // Remove garbage values
    int v0, v1, v2, v3;
    v0 = shared[threadIdx.x * 4 + 0];
    v1 = shared[threadIdx.x * 4 + 1];
    v2 = shared[threadIdx.x * 4 + 2];
    v3 = shared[threadIdx.x * 4 + 3];
    __syncthreads();

    if (v0 != garbage_val) {
        int write_idx = predicate[threadIdx.x * 4 + 0] - 1;
        shared[write_idx] = v0;
    }
    if (v1 != garbage_val) {
        int write_idx = predicate[threadIdx.x * 4 + 1] - 1;
        shared[write_idx] = v1;
    }
    if (v2 != garbage_val) {
        int write_idx = predicate[threadIdx.x * 4 + 2] - 1;
        shared[write_idx] = v2;
    }
    if (v3 != garbage_val) {
        int write_idx = predicate[threadIdx.x * 4 + 3] - 1;
        shared[write_idx] = v3;
    }
    __syncthreads();


}

void fix_image_gpu_handmade(Image& to_fix) { //rmm::device_uvector<int>& buffer, int size) {
    // Kernel is not finished :( But i did radix sort instead so it is worth it !!!
    std::cout << "GPU Handmade version has not been implemented yet !!!! Abort !!" << std::endl;
    return;
}
