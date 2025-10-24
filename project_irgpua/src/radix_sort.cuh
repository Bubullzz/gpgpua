#pragma once

#include "image.hh"

#include <rmm/device_uvector.hpp>

void radix_sort(std::vector<int>& array);
bool test_radix_sort();