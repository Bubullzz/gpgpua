#pragma once

#include "image.hh"

#include <rmm/device_uvector.hpp>

void fix_image_gpu_handmade(Image& to_fix);