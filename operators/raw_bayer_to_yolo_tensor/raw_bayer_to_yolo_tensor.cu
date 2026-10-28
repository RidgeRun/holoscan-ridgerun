// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "raw_bayer_to_yolo_tensor/raw_bayer_to_yolo_tensor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "operator_utils/tensor_utils.hpp"
#include "flow_trace/flow_trace.hpp"

namespace holoscan::ops {
namespace {

using holoscan::ops::forwarded_emit_options;
using holoscan::ops::invalid_tensor;
using holoscan::ops::kFloat32Dtype;
using holoscan::ops::kUInt16Dtype;
using holoscan::ops::same_dtype;
using holoscan::ops::shape_equals;

constexpr std::uint16_t kRaw10Mask = 0x03ffU;
constexpr int kRed = 0;
constexpr int kGreen = 1;
constexpr int kBlue = 2;

[[nodiscard]] std::size_t checked_bytes(std::int32_t width,
                                        std::int32_t height,
                                        std::size_t channels,
                                        std::size_t element_size) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("YOLO tensor dimensions must be positive");
  }
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (pixels > std::numeric_limits<std::size_t>::max() / channels ||
      pixels * channels > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error("YOLO tensor byte size overflows size_t");
  }
  return pixels * channels * element_size;
}

[[nodiscard]] holoscan::Error cuda_error(const char* operation, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string(operation) + " failed: " + cudaGetErrorString(status)};
}

__device__ __forceinline__ int bayer_color(int x, int y, int grid) {
  const bool odd_x = (x & 1) != 0;
  const bool odd_y = (y & 1) != 0;
  switch (grid) {
    case 0: return !odd_y ? (odd_x ? kGreen : kBlue) : (odd_x ? kRed : kGreen);
    case 1: return !odd_y ? (odd_x ? kGreen : kRed) : (odd_x ? kBlue : kGreen);
    case 2: return !odd_y ? (odd_x ? kRed : kGreen) : (odd_x ? kGreen : kBlue);
    default: return !odd_y ? (odd_x ? kBlue : kGreen) : (odd_x ? kGreen : kRed);
  }
}

// The baseline implementation scans the same 3x3 neighborhood once per color.
// This variant loads each neighborhood sample once and accumulates R/G/B together.
// It preserves the original bilinear interpolation and tensor contract while
// reducing mapped-host reads and repeated Bayer-color dispatch.
__device__ __forceinline__ void demosaic_rgb(const std::uint16_t* input,
                                             int center_x,
                                             int center_y,
                                             int width,
                                             int height,
                                             int optical_black,
                                             int raw_shift,
                                             int bayer_grid,
                                             float* channels) {
  float sums[3] = {0.0F, 0.0F, 0.0F};
  int counts[3] = {0, 0, 0};
#pragma unroll
  for (int delta_y = -1; delta_y <= 1; ++delta_y) {
    const int y = center_y + delta_y;
    if (y < 0 || y >= height) continue;
#pragma unroll
    for (int delta_x = -1; delta_x <= 1; ++delta_x) {
      const int x = center_x + delta_x;
      if (x < 0 || x >= width) continue;
      const int color = bayer_color(x, y, bayer_grid);
      const std::size_t index =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x;
      const int raw = static_cast<int>((input[index] >> raw_shift) & kRaw10Mask);
      sums[color] += static_cast<float>(max(0, raw - optical_black));
      ++counts[color];
    }
  }
  channels[kRed] = counts[kRed] == 0 ? 0.0F : sums[kRed] / counts[kRed];
  channels[kGreen] = counts[kGreen] == 0 ? 0.0F : sums[kGreen] / counts[kGreen];
  channels[kBlue] = counts[kBlue] == 0 ? 0.0F : sums[kBlue] / counts[kBlue];
}

__global__ void raw_bayer_to_yolo_kernel(const std::uint16_t* input,
                                         int source_width,
                                         int source_height,
                                         float* output,
                                         int network_width,
                                         int network_height,
                                         int resized_width,
                                         int resized_height,
                                         int padding_left,
                                         int padding_top,
                                         int optical_black,
                                         int raw_shift,
                                         int bayer_grid,
                                         float input_gain) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= network_width || y >= network_height) return;

  const std::size_t plane =
      static_cast<std::size_t>(network_width) * static_cast<std::size_t>(network_height);
  const std::size_t offset =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(network_width) + x;
  if (x < padding_left || x >= padding_left + resized_width ||
      y < padding_top || y >= padding_top + resized_height) {
    output[offset] = 0.0F;
    output[plane + offset] = 0.0F;
    output[2U * plane + offset] = 0.0F;
    return;
  }

  const float mapped_x =
      (static_cast<float>(x - padding_left) + 0.5F) * source_width / resized_width - 0.5F;
  const float mapped_y =
      (static_cast<float>(y - padding_top) + 0.5F) * source_height / resized_height - 0.5F;
  const int source_x = min(source_width - 1, max(0, __float2int_rn(mapped_x)));
  const int source_y = min(source_height - 1, max(0, __float2int_rn(mapped_y)));
  const float normalization = input_gain / static_cast<float>(1023 - optical_black);
  float channels[3];
  demosaic_rgb(input, source_x, source_y, source_width, source_height, optical_black,
               raw_shift, bayer_grid, channels);
  output[offset] = fminf(1.0F, channels[kRed] * normalization);
  output[plane + offset] = fminf(1.0F, channels[kGreen] * normalization);
  output[2U * plane + offset] = fminf(1.0F, channels[kBlue] * normalization);
}

}  // namespace

RawBayerToYoloTensorOp::RawBayerToYoloTensorOp(std::int32_t source_width,
                                               std::int32_t source_height,
                                               std::int32_t network_width,
                                               std::int32_t network_height,
                                               std::int32_t optical_black,
                                               std::int32_t bayer_grid,
                                               std::int32_t raw_shift,
                                               float input_gain)
    : source_width_(source_width),
      source_height_(source_height),
      network_width_(network_width),
      network_height_(network_height),
      optical_black_(optical_black),
      bayer_grid_(bayer_grid),
      raw_shift_(raw_shift),
      input_gain_(input_gain) {
  static_cast<void>(checked_bytes(source_width_, source_height_, 1U, sizeof(std::uint16_t)));
  static_cast<void>(checked_bytes(network_width_, network_height_, 3U, sizeof(float)));
  if ((source_width_ & 1) != 0 || (source_height_ & 1) != 0 ||
      optical_black_ < 0 || optical_black_ >= 1023 || bayer_grid_ < 0 ||
      bayer_grid_ > 3 || raw_shift_ < 0 || raw_shift_ > 6 ||
      !std::isfinite(input_gain_) || input_gain_ <= 0.0F) {
    throw std::invalid_argument("invalid fused RG10-to-YOLO preprocessing configuration");
  }
  const float scale = std::min(static_cast<float>(network_width_) / source_width_,
                               static_cast<float>(network_height_) / source_height_);
  resized_width_ = std::max(1, static_cast<std::int32_t>(std::lround(source_width_ * scale)));
  resized_height_ = std::max(1, static_cast<std::int32_t>(std::lround(source_height_ * scale)));
  padding_left_ = (network_width_ - resized_width_) / 2;
  padding_top_ = (network_height_ - resized_height_) / 2;
}

void RawBayerToYoloTensorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = {
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .dtype = kUInt16Dtype,
          .rank = 3U,
      }});
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kFloat32Dtype,
              .rank = 4U,
          },
          .bounds = holoscan::tensor_bounds(
              checked_bytes(network_width_, network_height_, 3U, sizeof(float))),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract RawBayerToYoloTensorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

holoscan::expected<void, holoscan::Error> RawBayerToYoloTensorOp::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported, "CUDA stream is unavailable"});
  }
  auto sample = input.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  const std::uint64_t trace_frame_id = sample->metadata.frame_id;
  const std::int64_t trace_receive_us =
      holoscan::ops::FlowTrace::now_microseconds();
  const std::array<std::int64_t, 3U> expected_shape{source_height_, source_width_, 1};
  const holoscan::Tensor& tensor = sample->data;
  if (tensor.device().device_type != kDLCUDA ||
      !same_dtype(tensor.dtype(), kUInt16Dtype) ||
      !shape_equals(tensor, expected_shape) || !tensor.is_contiguous() ||
      tensor.data() == nullptr) {
    return holoscan::make_unexpected(invalid_tensor(
        "RawBayerToYoloTensorOp expects contiguous CUDA uint16 [height,width,1] RG10 input"));
  }

  const std::array<std::int64_t, 4U> output_shape{1, 3, network_height_, network_width_};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = output_shape, .dtype = kFloat32Dtype});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());

  constexpr dim3 block{16U, 16U, 1U};
  const dim3 grid{static_cast<unsigned>((network_width_ + 15) / 16),
                  static_cast<unsigned>((network_height_ + 15) / 16), 1U};
  raw_bayer_to_yolo_kernel<<<grid, block, 0U, stream>>>(
      tensor.data_as<std::uint16_t>(), source_width_, source_height_, writer->data_as<float>(),
      network_width_, network_height_, resized_width_, resized_height_, padding_left_, padding_top_,
      optical_black_, raw_shift_, bayer_grid_, input_gain_);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("fused RG10-to-YOLO kernel", status));
  }
  auto committed = std::move(*writer).commit();
  if (!committed) return holoscan::make_unexpected(std::move(committed).error());
  auto emitted = output.emit(std::move(*loan), forwarded_emit_options(sample->metadata));
  if (!emitted) return emitted;
  holoscan::ops::FlowTrace::instance().record(
      trace_frame_id, "preprocess", trace_receive_us,
      holoscan::ops::FlowTrace::now_microseconds());
  return emitted;
}

}  // namespace holoscan::ops
