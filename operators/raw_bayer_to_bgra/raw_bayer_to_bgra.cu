// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "raw_bayer_to_bgra/raw_bayer_to_bgra.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "raw_bayer_to_bgra/exportable_holomq_pool.hpp"
#include "flow_trace/flow_trace.hpp"
#include "operator_utils/tensor_utils.hpp"

namespace holoscan::ops {
namespace {

constexpr std::uint16_t kRaw10Mask = 0x03ffU;
constexpr int kRed = 0;
constexpr int kGreen = 1;
constexpr int kBlue = 2;
constexpr std::uint32_t kExportableOutputSlots = 16U;

[[nodiscard]] holoscan::Error cuda_error(const char* operation, cudaError_t status) {
  return holoscan::Error{
      holoscan::ErrorCode::kFailure,
      std::string(operation) + " failed: " + cudaGetErrorString(status)};
}

[[nodiscard]] std::size_t checked_image_bytes(std::int32_t width,
                                              std::int32_t height,
                                              std::size_t bytes_per_pixel) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("RawBayerToBgraOp dimensions must be positive");
  }
  const auto unsigned_width = static_cast<std::size_t>(width);
  const auto unsigned_height = static_cast<std::size_t>(height);
  if (unsigned_width > std::numeric_limits<std::size_t>::max() / unsigned_height) {
    throw std::overflow_error("RawBayerToBgraOp pixel count overflows size_t");
  }
  const std::size_t pixels = unsigned_width * unsigned_height;
  if (pixels > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
    throw std::overflow_error("RawBayerToBgraOp image byte size overflows size_t");
  }
  return pixels * bytes_per_pixel;
}

__device__ __forceinline__ int bayer_color(int x, int y, int grid) {
  const bool odd_x = (x & 1) != 0;
  const bool odd_y = (y & 1) != 0;
  switch (grid) {
    case 0:  // BGGR
      if (!odd_y) {
        return odd_x ? kGreen : kBlue;
      }
      return odd_x ? kRed : kGreen;
    case 1:  // RGGB, used by V4L2 RG10 on IMX274
      if (!odd_y) {
        return odd_x ? kGreen : kRed;
      }
      return odd_x ? kBlue : kGreen;
    case 2:  // GRBG
      if (!odd_y) {
        return odd_x ? kRed : kGreen;
      }
      return odd_x ? kGreen : kBlue;
    default:  // GBRG
      if (!odd_y) {
        return odd_x ? kBlue : kGreen;
      }
      return odd_x ? kGreen : kRed;
  }
}

__device__ __forceinline__ int decode_raw10(std::uint16_t value, int raw_shift) {
  return static_cast<int>((value >> raw_shift) & kRaw10Mask);
}

__global__ void accumulate_channels_kernel(const std::uint16_t* input,
                                           std::size_t pixels,
                                           int width,
                                           int optical_black,
                                           int raw_shift,
                                           int bayer_grid,
                                           unsigned long long* sums) {
  constexpr unsigned int kThreads = 256U;
  __shared__ unsigned long long red[kThreads];
  __shared__ unsigned long long green[kThreads];
  __shared__ unsigned long long blue[kThreads];

  const unsigned int lane = threadIdx.x;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + lane;
  unsigned long long value = 0ULL;
  int color = kGreen;
  if (index < pixels) {
    const int raw = decode_raw10(input[index], raw_shift);
    value = static_cast<unsigned long long>(
        raw > optical_black ? raw - optical_black : 0);
    const int x = static_cast<int>(index % static_cast<std::size_t>(width));
    const int y = static_cast<int>(index / static_cast<std::size_t>(width));
    color = bayer_color(x, y, bayer_grid);
  }
  red[lane] = color == kRed ? value : 0ULL;
  green[lane] = color == kGreen ? value : 0ULL;
  blue[lane] = color == kBlue ? value : 0ULL;
  __syncthreads();

  for (unsigned int offset = kThreads / 2U; offset > 0U; offset >>= 1U) {
    if (lane < offset) {
      red[lane] += red[lane + offset];
      green[lane] += green[lane + offset];
      blue[lane] += blue[lane + offset];
    }
    __syncthreads();
  }
  if (lane == 0U) {
    atomicAdd(&sums[kRed], red[0]);
    atomicAdd(&sums[kGreen], green[0]);
    atomicAdd(&sums[kBlue], blue[0]);
  }
}

__global__ void calculate_gains_kernel(const unsigned long long* sums,
                                       std::uint64_t red_count,
                                       std::uint64_t green_count,
                                       std::uint64_t blue_count,
                                       float* gains) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  const float red_average =
      static_cast<float>(sums[kRed]) / static_cast<float>(red_count);
  const float green_average =
      static_cast<float>(sums[kGreen]) / static_cast<float>(green_count);
  const float blue_average =
      static_cast<float>(sums[kBlue]) / static_cast<float>(blue_count);

  constexpr float kMinimumAverage = 1.0F;
  constexpr float kMinimumGain = 0.25F;
  constexpr float kMaximumGain = 4.0F;
  gains[kRed] =
      fminf(kMaximumGain,
            fmaxf(kMinimumGain, green_average / fmaxf(red_average, kMinimumAverage)));
  gains[kGreen] = 1.0F;
  gains[kBlue] =
      fminf(kMaximumGain,
            fmaxf(kMinimumGain, green_average / fmaxf(blue_average, kMinimumAverage)));
}

__device__ __forceinline__ float interpolated_channel(const std::uint16_t* input,
                                                      int x,
                                                      int y,
                                                      int width,
                                                      int height,
                                                      int optical_black,
                                                      int raw_shift,
                                                      int bayer_grid,
                                                      int requested_color,
                                                      const float* gains) {
  float sum = 0.0F;
  int count = 0;
  for (int delta_y = -1; delta_y <= 1; ++delta_y) {
    const int sample_y = y + delta_y;
    if (sample_y < 0 || sample_y >= height) {
      continue;
    }
    for (int delta_x = -1; delta_x <= 1; ++delta_x) {
      const int sample_x = x + delta_x;
      if (sample_x < 0 || sample_x >= width ||
          bayer_color(sample_x, sample_y, bayer_grid) != requested_color) {
        continue;
      }
      const std::size_t index =
          static_cast<std::size_t>(sample_y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(sample_x);
      const int raw = decode_raw10(input[index], raw_shift);
      const int corrected = raw > optical_black ? raw - optical_black : 0;
      sum += static_cast<float>(corrected) * gains[requested_color];
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<float>(count) : 0.0F;
}

__device__ __forceinline__ std::uint8_t scale_raw10(
    float value, int optical_black, float display_gain, float gamma) {
  const float usable_range = static_cast<float>(1023 - optical_black);
  const float linear = fminf(1.0F, fmaxf(0.0F, value * display_gain / usable_range));
  const float encoded = gamma == 1.0F ? linear : powf(linear, 1.0F / gamma);
  return static_cast<std::uint8_t>(encoded * 255.0F + 0.5F);
}

__global__ void demosaic_to_bgra_kernel(const std::uint16_t* input,
                                        std::uint8_t* output,
                                        int width,
                                        int height,
                                        int optical_black,
                                        int raw_shift,
                                        int bayer_grid,
                                        float display_gain,
                                        float gamma,
                                        const float* gains) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }

  const float red = interpolated_channel(
      input, x, y, width, height, optical_black, raw_shift, bayer_grid, kRed, gains);
  const float green = interpolated_channel(
      input, x, y, width, height, optical_black, raw_shift, bayer_grid, kGreen, gains);
  const float blue = interpolated_channel(
      input, x, y, width, height, optical_black, raw_shift, bayer_grid, kBlue, gains);
  const std::size_t output_offset =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
       static_cast<std::size_t>(x)) *
      4U;
  output[output_offset] = scale_raw10(blue, optical_black, display_gain, gamma);
  output[output_offset + 1U] = scale_raw10(green, optical_black, display_gain, gamma);
  output[output_offset + 2U] = scale_raw10(red, optical_black, display_gain, gamma);
  output[output_offset + 3U] = 255U;
}

}  // namespace

RawBayerToBgraOp::RawBayerToBgraOp(std::int32_t width,
                                   std::int32_t height,
                                   std::int32_t optical_black,
                                   std::int32_t bayer_grid,
                                   std::int32_t cuda_device,
                                   std::int32_t raw_shift,
                                   float display_gain,
                                   float gamma,
                                   std::optional<std::array<float, 3U>> fixed_white_balance,
                                   bool require_exportable_output)
    : width_(width),
      height_(height),
      optical_black_(optical_black),
      bayer_grid_(bayer_grid),
      cuda_device_(cuda_device),
      raw_shift_(raw_shift),
      display_gain_(display_gain),
      gamma_(gamma),
      fixed_white_balance_(std::move(fixed_white_balance)),
      require_exportable_output_(require_exportable_output),
      input_bytes_(checked_image_bytes(width, height, sizeof(std::uint16_t))),
      output_bytes_(checked_image_bytes(width, height, 4U)) {
  if ((width_ & 1) != 0 || (height_ & 1) != 0) {
    throw std::invalid_argument("RawBayerToBgraOp requires even image dimensions");
  }
  if (optical_black_ < 0 || optical_black_ >= 1023) {
    throw std::invalid_argument("optical black must be in the range [0,1022]");
  }
  if (bayer_grid_ < 0 || bayer_grid_ > 3) {
    throw std::invalid_argument("bayer grid must be 0=BGGR, 1=RGGB, 2=GRBG, or 3=GBRG");
  }
  if (cuda_device_ < 0) {
    throw std::invalid_argument("CUDA device must be nonnegative");
  }
  if (raw_shift_ < 0 || raw_shift_ > 6) {
    throw std::invalid_argument("RAW10 bit shift must be in the range [0,6]");
  }
  if (!std::isfinite(display_gain_) || display_gain_ <= 0.0F || display_gain_ > 64.0F) {
    throw std::invalid_argument("display gain must be finite and in the range (0,64]");
  }
  if (!std::isfinite(gamma_) || gamma_ <= 0.0F || gamma_ > 5.0F) {
    throw std::invalid_argument("gamma must be finite and in the range (0,5]");
  }
  if (fixed_white_balance_) {
    for (const float gain : *fixed_white_balance_) {
      if (!std::isfinite(gain) || gain <= 0.0F || gain > 16.0F) {
        throw std::invalid_argument(
            "fixed white-balance gains must be finite and in the range (0,16]");
      }
    }
  }
  const std::uint64_t pixels =
      static_cast<std::uint64_t>(width_) * static_cast<std::uint64_t>(height_);
  red_count_ = pixels / 4U;
  blue_count_ = pixels / 4U;
  green_count_ = pixels / 2U;
}

RawBayerToBgraOp::~RawBayerToBgraOp() { cleanup(); }

void RawBayerToBgraOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt16Dtype,
              .rank = 3U,
          },
      });
  holoscan::TensorOutputSpec output_spec{
      .representation = {
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .dtype = kUInt8Dtype,
          .rank = 3U,
      },
      .bounds = holoscan::tensor_bounds(output_bytes_),
      // Holoscan SDK (HSDK) 5 requires a RuntimePool declaration to materialize Output<Tensor>::emit's
      // port binding. Exportable mode does not allocate from this pool; it emits the Tensor
      // loaned from holomq_output_pool_ below.
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
  if (require_exportable_output_) {
    output_spec.transfer = holoscan::TensorTransferPolicy{
        .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kRequireExportable,
    };
  }
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(output_spec);
}

holoscan::Contract RawBayerToBgraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

void RawBayerToBgraOp::start() {
  cleanup();
  first_frame_logged_ = false;
  cudaError_t status = cudaSetDevice(cuda_device_);
  if (status == cudaSuccess && require_exportable_output_) {
    status = cudaStreamCreateWithFlags(&owned_output_stream_, cudaStreamNonBlocking);
  }
  if (status == cudaSuccess && require_exportable_output_) {
    try {
      holomq_output_pool_ = std::make_unique<ExportableHoloMQPool>(
          output_bytes_, kExportableOutputSlots, cuda_device_);
    } catch (const std::exception& exception) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kFailure,
          std::string{"RawBayerToBgraOp exportable output initialization failed: "} +
              exception.what());
    }
  }
  if (status == cudaSuccess && !fixed_white_balance_) {
    status = cudaMalloc(reinterpret_cast<void**>(&channel_sums_),
                        3U * sizeof(unsigned long long));
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(reinterpret_cast<void**>(&channel_gains_), 3U * sizeof(float));
  }
  if (status == cudaSuccess && fixed_white_balance_) {
    status = cudaMemcpy(channel_gains_,
                        fixed_white_balance_->data(),
                        fixed_white_balance_->size() * sizeof(float),
                        cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    cleanup();
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure,
        std::string("RawBayerToBgraOp CUDA initialization failed: ") +
            cudaGetErrorString(status));
  }
}

void RawBayerToBgraOp::stop() {
  if (require_exportable_output_) {
    // External publishers are torn down after operator stop. Keep the cold-bound HoloMQ leases
    // alive until this operator is destroyed so their final queued/imported claims can retire.
    if (holomq_output_stream_ != nullptr) {
      static_cast<void>(cudaStreamSynchronize(holomq_output_stream_));
    }
    return;
  }
  cleanup();
}

holoscan::expected<void, holoscan::Error> RawBayerToBgraOp::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream =
      require_exportable_output_ ? owned_output_stream_ : context.cuda_stream().get();
  if (stream == nullptr || channel_gains_ == nullptr ||
      (!fixed_white_balance_ && channel_sums_ == nullptr)) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotReady,
                        "Raw Bayer processing CUDA resources are unavailable"});
  }

  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }
  const std::uint64_t frame_id = sample->metadata.frame_id;
  const std::int64_t trace_receive_us = FlowTrace::now_microseconds();
  const std::array<std::int64_t, 3U> input_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      1,
  };
  const holoscan::Tensor& tensor = sample->data;
  const std::uint16_t* input_data = tensor.data_as<std::uint16_t>();
  if (tensor.device().device_type != kDLCUDA ||
      tensor.device().device_id != cuda_device_ ||
      !same_dtype(tensor.dtype(), kUInt16Dtype) ||
      !shape_equals(tensor, input_shape) ||
      !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(input_bytes_) ||
      input_data == nullptr) {
    return holoscan::make_unexpected(invalid_tensor(
        "RawBayerToBgraOp expects contiguous CUDA uint16 [height,width,1] input"));
  }

  const std::array<std::int64_t, 3U> output_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      4,
  };
  std::optional<holoscan::TensorOutputLoan> runtime_loan;
  std::optional<holoscan::TensorOutputWriteGuard> runtime_writer;
  holoscan::Tensor exportable_tensor;
  std::uint8_t* output_data = nullptr;
  if (require_exportable_output_) {
    if (!holomq_output_pool_) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kNotReady,
                          "HoloMQ exportable output pool is unavailable"});
    }
    auto wrapped = holomq_output_pool_->acquire_tensor(output_shape, kUInt8Dtype);
    if (!wrapped) {
      return holoscan::make_unexpected(std::move(wrapped).error());
    }
    exportable_tensor = std::move(*wrapped);
    auto mutable_data = exportable_tensor.mutable_data();
    if (!mutable_data || *mutable_data == nullptr ||
        exportable_tensor.nbytes() != static_cast<std::int64_t>(output_bytes_)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "invalid HoloMQ exportable BGRA output allocation"});
    }
    output_data = static_cast<std::uint8_t*>(*mutable_data);
  } else {
    auto loan = output.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = output_shape, .dtype = kUInt8Dtype});
    if (!loan) {
      holoscan::Error error = std::move(loan).error();
      error.message = "RawBayerToBgraOp output.allocate_tensor failed: " + error.message;
      return holoscan::make_unexpected(std::move(error));
    }
    runtime_loan.emplace(std::move(*loan));
    auto writer = runtime_loan->write(context.cuda_stream());
    if (!writer) {
      holoscan::Error error = std::move(writer).error();
      error.message = "RawBayerToBgraOp output loan write failed: " + error.message;
      return holoscan::make_unexpected(std::move(error));
    }
    runtime_writer.emplace(std::move(*writer));
    if (runtime_writer->data() == nullptr || runtime_writer->byte_size() != output_bytes_) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "invalid raw Bayer BGRA output allocation"});
    }
    output_data = runtime_writer->data_as<std::uint8_t>();
  }

  cudaError_t status = cudaSuccess;
  if (!fixed_white_balance_) {
    status = cudaMemsetAsync(channel_sums_, 0, 3U * sizeof(unsigned long long), stream);
    if (status != cudaSuccess) {
      return holoscan::make_unexpected(cuda_error("white-balance accumulator reset", status));
    }

    constexpr unsigned int kReductionThreads = 256U;
    const std::size_t pixels =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    const unsigned int reduction_blocks = static_cast<unsigned int>(
        (pixels + kReductionThreads - 1U) / kReductionThreads);
    accumulate_channels_kernel<<<reduction_blocks, kReductionThreads, 0U, stream>>>(
        input_data,
        pixels,
        width_,
        optical_black_,
        raw_shift_,
        bayer_grid_,
        channel_sums_);
    calculate_gains_kernel<<<1U, 1U, 0U, stream>>>(
        channel_sums_, red_count_, green_count_, blue_count_, channel_gains_);
  }

  if (!first_frame_logged_) {
    std::array<float, 3U> gains = fixed_white_balance_.value_or(std::array<float, 3U>{});
    if (!fixed_white_balance_) {
      status = cudaMemcpyAsync(gains.data(),
                               channel_gains_,
                               gains.size() * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               stream);
      if (status == cudaSuccess) {
        status = cudaStreamSynchronize(stream);
      }
      if (status != cudaSuccess) {
        return holoscan::make_unexpected(
            cuda_error("first-frame white-balance diagnostic", status));
      }
    }
    std::fprintf(stdout,
                 "[raw_bayer_to_bgra] first RG10 frame: %s WB gains R=%.3f G=%.3f B=%.3f\n",
                 fixed_white_balance_ ? "fixed" : "gray-world",
                 gains[0], gains[1], gains[2]);
    first_frame_logged_ = true;
  }

  constexpr dim3 block{16U, 16U, 1U};
  const dim3 grid{
      static_cast<unsigned int>((width_ + static_cast<int>(block.x) - 1) /
                                static_cast<int>(block.x)),
      static_cast<unsigned int>((height_ + static_cast<int>(block.y) - 1) /
                                static_cast<int>(block.y)),
      1U,
  };
  demosaic_to_bgra_kernel<<<grid, block, 0U, stream>>>(
      input_data,
      output_data,
      width_,
      height_,
      optical_black_,
      raw_shift_,
      bayer_grid_,
      display_gain_,
      gamma_,
      channel_gains_);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return holoscan::make_unexpected(
        cuda_error("raw correction/white balance/demosaic kernel launch", status));
  }

  auto emit_options = forwarded_emit_options(sample->metadata);
  holoscan::expected<void, holoscan::Error> emitted;
  if (require_exportable_output_) {
    if (!exportable_tensor.set_readiness_stream(stream)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "recording HoloMQ BGRA output readiness failed"});
    }
    holomq_output_stream_ = stream;
    emitted = output.emit(std::move(exportable_tensor), std::move(emit_options));
  } else {
    auto committed = std::move(*runtime_writer).commit();
    if (!committed) {
      holoscan::Error error = std::move(committed).error();
      error.message = "RawBayerToBgraOp output commit failed: " + error.message;
      return holoscan::make_unexpected(std::move(error));
    }
    emitted = output.emit(std::move(*runtime_loan), std::move(emit_options));
  }
  if (!emitted) {
    holoscan::Error error = std::move(emitted).error();
    error.message = "RawBayerToBgraOp output emit failed: " + error.message;
    return holoscan::make_unexpected(std::move(error));
  }
  FlowTrace::instance().record(frame_id,
                                    "convert",
                                    trace_receive_us,
                                    FlowTrace::now_microseconds());
  return {};
}

void RawBayerToBgraOp::cleanup() noexcept {
  if (cuda_device_ >= 0) {
    static_cast<void>(cudaSetDevice(cuda_device_));
  }
  if (holomq_output_stream_ != nullptr) {
    static_cast<void>(cudaStreamSynchronize(holomq_output_stream_));
    holomq_output_stream_ = nullptr;
  }
  holomq_output_pool_.reset();
  if (owned_output_stream_ != nullptr) {
    static_cast<void>(cudaStreamDestroy(owned_output_stream_));
    owned_output_stream_ = nullptr;
  }
  if (channel_gains_ != nullptr) {
    static_cast<void>(cudaFree(channel_gains_));
    channel_gains_ = nullptr;
  }
  if (channel_sums_ != nullptr) {
    static_cast<void>(cudaFree(channel_sums_));
    channel_sums_ = nullptr;
  }
}

}  // namespace holoscan::ops
