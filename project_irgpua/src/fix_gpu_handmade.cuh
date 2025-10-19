#pragma once

#include "image.hh"

#include <rmm/device_uvector.hpp>

void fix_image_gpu_handmade(rmm::device_uvector<int>& buffer, int size);