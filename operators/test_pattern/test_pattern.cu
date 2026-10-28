// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <array>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "test_pattern/test_pattern.hpp"

namespace holoscan::ops {
namespace {

__global__ void color_bars_kernel(std::uint8_t* output, const std::int32_t width,
                                  const std::int32_t height, const std::uint64_t frame_id) {
  const std::int32_t x = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t y = static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }

  const std::int32_t bar = (x * 6) / width;
  const std::int32_t phase = static_cast<std::int32_t>((frame_id / 30U) % 6U);
  const std::int32_t color = (bar + phase) % 6;
  std::uint8_t blue = 0U;
  std::uint8_t green = 0U;
  std::uint8_t red = 0U;
  switch (color) {
    case 0:
      blue = 255U;
      green = 255U;
      red = 255U;
      break;
    case 1:
      blue = 0U;
      green = 255U;
      red = 255U;
      break;
    case 2:
      blue = 255U;
      green = 255U;
      red = 0U;
      break;
    case 3:
      blue = 0U;
      green = 255U;
      red = 0U;
      break;
    case 4:
      blue = 255U;
      green = 0U;
      red = 255U;
      break;
    default:
      blue = 0U;
      green = 0U;
      red = 255U;
      break;
  }
  const std::size_t output_index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)) *
                                   4U;
  output[output_index] = blue;
  output[output_index + 1U] = green;
  output[output_index + 2U] = red;
  output[output_index + 3U] = 255U;
}

__global__ void checkerboard_kernel(std::uint8_t* output, const std::int32_t width,
                                    const std::int32_t height, const std::uint64_t frame_id) {
  const std::int32_t x = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t y = static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }

  constexpr std::int32_t kCell = 48;
  const bool light = ((x / kCell + y / kCell + static_cast<std::int32_t>(frame_id / 15U)) & 1) != 0;
  const std::uint8_t blue = light ? 40U : 220U;
  const std::uint8_t green = light ? 180U : 40U;
  const std::uint8_t red = light ? 255U : 20U;
  const std::size_t output_index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)) *
                                   4U;
  output[output_index] = blue;
  output[output_index + 1U] = green;
  output[output_index + 2U] = red;
  output[output_index + 3U] = 255U;
}

[[nodiscard]] holoscan::Error cuda_error(const char* operation, const cudaError_t status) {
  return holoscan::Error{holoscan::ErrorCode::kFailure,
                         std::string(operation) + " failed: " + cudaGetErrorString(status)};
}

}  // namespace

TestPatternOp::TestPatternOp(const std::int32_t width, const std::int32_t height,
                             const std::int32_t fps, const TestPatternKind pattern)
    : width_(width), height_(height), pattern_(pattern) {
  if (width_ <= 0 || height_ <= 0 || fps <= 0) {
    throw std::invalid_argument("TestPatternOp width, height, and fps must be positive");
  }
  const auto width_u64 = static_cast<std::uint64_t>(width_);
  const auto height_u64 = static_cast<std::uint64_t>(height_);
  if (width_u64 > std::numeric_limits<std::uint64_t>::max() / height_u64 ||
      width_u64 * height_u64 > std::numeric_limits<std::uint64_t>::max() / 4U) {
    throw std::overflow_error("TestPatternOp frame size overflows uint64_t");
  }
  if (fps > 1'000'000'000) {
    throw std::invalid_argument("TestPatternOp fps exceeds nanosecond clock resolution");
  }
  period_ = std::chrono::nanoseconds{1'000'000'000LL / fps};
}

void TestPatternOp::setup(holoscan::OperatorSpec& spec) {
  const auto frame_bytes =
      static_cast<std::uint64_t>(width_) * static_cast<std::uint64_t>(height_) * 4U;
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 3U,
          },
          .bounds = holoscan::tensor_bounds(frame_bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract TestPatternOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period_});
  return result;
}

holoscan::expected<void, holoscan::Error> TestPatternOp::compute(
    holoscan::ExecutionContext& context) {
  const auto stream = context.cuda_stream();
  if (stream.get() == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotSupported,
                                                     "TestPatternOp requires a CUDA stream"});
  }

  const std::array<std::int64_t, 3U> shape{static_cast<std::int64_t>(height_),
                                           static_cast<std::int64_t>(width_), 4};
  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(stream);
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->data() == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "TestPatternOp received an empty output loan"});
  }

  constexpr dim3 block{16U, 16U, 1U};
  const dim3 grid{
      static_cast<unsigned int>((width_ + static_cast<std::int32_t>(block.x) - 1) /
                                static_cast<std::int32_t>(block.x)),
      static_cast<unsigned int>((height_ + static_cast<std::int32_t>(block.y) - 1) /
                                static_cast<std::int32_t>(block.y)),
      1U,
  };
  if (pattern_ == TestPatternKind::kColorBars) {
    color_bars_kernel<<<grid, block, 0U, stream.get()>>>(writer->data_as<std::uint8_t>(), width_,
                                                         height_, frame_id_);
  } else {
    checkerboard_kernel<<<grid, block, 0U, stream.get()>>>(writer->data_as<std::uint8_t>(), width_,
                                                           height_, frame_id_);
  }
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return holoscan::make_unexpected(cuda_error("TestPatternOp kernel launch", status));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  holoscan::EmitOptions options;
  options.frame_id = frame_id_++;
  return output.emit(std::move(*loan), options);
}

}  // namespace holoscan::ops
