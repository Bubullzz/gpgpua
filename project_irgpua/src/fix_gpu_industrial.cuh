#pragma once

#include "image.hh"

#include <rmm/device_uvector.hpp>

void fix_image_gpu_industrial(Image& to_fix);
void sort_gpu_industrial(std::vector<Image::ToSort>& to_sort, int nb_images);