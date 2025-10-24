#pragma once

#include "image.hh"

#include <assert.h>
#include <raft/core/device_span.hpp>

#include <cub/cub.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>


void radix_sort(rmm::device_uvector<int>& in, rmm::device_uvector<int>& out, int max_value, cudaStream_t stream);
bool test_radix_sort();