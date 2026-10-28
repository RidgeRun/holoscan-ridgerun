// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "drm_display_kernel.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::ops {
namespace {

__device__ __forceinline__ std::uint8_t clamp_byte(int value) {
  const int nonnegative = value < 0 ? 0 : value;
  return static_cast<std::uint8_t>(nonnegative > 255 ? 255 : nonnegative);
}

__device__ __forceinline__ void write_pixel(std::uint8_t* output,
                                             std::size_t output_pitch,
                                             std::int32_t y,
                                             int luma,
                                             std::size_t pixel_x,
                                             int u,
                                             int v) {
  const int c = luma > 16 ? luma - 16 : 0;
  const int red = (298 * c + 409 * v + 128) >> 8;
  const int green = (298 * c - 100 * u - 208 * v + 128) >> 8;
  const int blue = (298 * c + 516 * u + 128) >> 8;
  std::uint8_t* pixel = output + static_cast<std::size_t>(y) * output_pitch +
                        pixel_x * 4U;
  pixel[0] = clamp_byte(blue);
  pixel[1] = clamp_byte(green);
  pixel[2] = clamp_byte(red);
  pixel[3] = 255U;
}

__global__ void yuyv_to_bgra_kernel(const std::uint8_t* input,
                                    std::uint8_t* output,
                                    std::size_t output_pitch,
                                    std::int32_t width,
                                    std::int32_t height) {
  const std::int32_t pair_x =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t y =
      static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (pair_x >= width / 2 || y >= height) {
    return;
  }

  const std::size_t pair_index =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(width / 2) +
      static_cast<std::size_t>(pair_x);
  const std::size_t input_offset = pair_index * 4U;
  const int y0 = static_cast<int>(input[input_offset]);
  const int u = static_cast<int>(input[input_offset + 1U]) - 128;
  const int y1 = static_cast<int>(input[input_offset + 2U]);
  const int v = static_cast<int>(input[input_offset + 3U]) - 128;

  const std::size_t first_pixel = static_cast<std::size_t>(pair_x) * 2U;
  write_pixel(output, output_pitch, y, y0, first_pixel, u, v);
  write_pixel(output, output_pitch, y, y1, first_pixel + 1U, u, v);
}

}  // namespace

cudaError_t launch_yuyv_to_bgra(const std::uint8_t* input,
                                std::uint8_t* output,
                                std::size_t output_pitch,
                                std::int32_t width,
                                std::int32_t height,
                                cudaStream_t stream) noexcept {
  constexpr dim3 block{16U, 16U, 1U};
  const dim3 grid{
      static_cast<unsigned int>((width / 2 + static_cast<std::int32_t>(block.x) - 1) /
                                static_cast<std::int32_t>(block.x)),
      static_cast<unsigned int>((height + static_cast<std::int32_t>(block.y) - 1) /
                                static_cast<std::int32_t>(block.y)),
      1U};
  yuyv_to_bgra_kernel<<<grid, block, 0U, stream>>>(
      input, output, output_pitch, width, height);
  return cudaGetLastError();
}

}  // namespace holoscan::ops
