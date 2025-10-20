#pragma once
#include "fix_cpu.cuh"
#include "fix_gpu_handmade.cuh"
#include "fix_gpu_industrial.cuh"

#include <vector>
#include <sstream>
#include <numeric>

enum class ProcessingMode {
    CPU,
    GPU_Handmade,
    GPU_Industrial
};

// Compile-time dispatch to the correct processing mode so no performance loss hihi ^^
template <ProcessingMode mode>
void fix_image(Image& to_fix)
{
    if constexpr (mode == ProcessingMode::CPU) {
        fix_image_cpu(to_fix);
    } else if constexpr (mode == ProcessingMode::GPU_Handmade) {
        fix_image_gpu_handmade(to_fix);
    } else { // GPU_Industrial
        fix_image_gpu_industrial(to_fix);
    }
}

template <ProcessingMode mode>
void compute_total(std::vector<Image>& images, int nb_images)
{
    if constexpr (mode == ProcessingMode::CPU) {
        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            auto& image = images[i];
            const int image_size = image.width * image.height;
            image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
        }
    } else if constexpr (mode == ProcessingMode::GPU_Handmade) {
        return; // Already computed in fix_image_gpu_handmade
    } else { // GPU_Industrial
        return; // Already computed in fix_image_gpu_industrial
    }
}

template <ProcessingMode mode>
void print_mode()
{
    if constexpr (mode == ProcessingMode::CPU) {
        std::cout << "Processing mode: CPU" << std::endl;
    } else if constexpr (mode == ProcessingMode::GPU_Handmade) {
        std::cout << "Processing mode: GPU (Handmade)" << std::endl;
    } else { // GPU_Industrial
        std::cout << "Processing mode: GPU (Industrial)" << std::endl;
    }
}