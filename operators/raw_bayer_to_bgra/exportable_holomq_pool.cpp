// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "raw_bayer_to_bgra/exportable_holomq_pool.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holomq/buffer.hpp>
#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/context.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/errors.hpp>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/memory/tensor_storage.hpp>

// Holoscan SDK (HSDK) 5 keeps this exportable-memory contract source-private. Repeating its ABI declaration here
// confines the SDK dependency to this adapter while still allowing the public Tensor publisher to
// identify the backing as HoloMQ-exportable.
namespace holoscan::channel_internal {
class HoloMQExportableTensorMemoryReference : public TensorMemoryReference {
 public:
  [[nodiscard]] virtual expected<holomq::BufferHandle, Error> export_handle() noexcept = 0;
};
}  // namespace holoscan::channel_internal

namespace holoscan::ops {
namespace {

struct ExportSlot {
  explicit ExportSlot(holomq::Buffer value) : buffer(std::move(value)) {}

  holomq::Buffer buffer;
  std::mutex mutex;
  std::atomic_bool in_use{false};
  bool first_generation{true};
  cudaStream_t readiness_stream{};
};

class ExportableSlotReference final
    : public holoscan::channel_internal::HoloMQExportableTensorMemoryReference {
 public:
  explicit ExportableSlotReference(std::shared_ptr<ExportSlot> slot) : slot_(std::move(slot)) {}

  void retain_tensor_handle() noexcept override {
    handle_count_.fetch_add(1U, std::memory_order_relaxed);
  }

  void release_tensor_handle() noexcept override {
    const std::uint32_t previous = handle_count_.fetch_sub(1U, std::memory_order_acq_rel);
    if (previous == 1U) {
      slot_->in_use.store(false, std::memory_order_release);
    }
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    if (stream == nullptr) {
      return false;
    }
    const std::lock_guard lock(slot_->mutex);
    if (!slot_->buffer.enqueue_ready_set(stream)) {
      return false;
    }
    slot_->readiness_stream = stream;
    return true;
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    const std::lock_guard lock(slot_->mutex);
    return slot_->readiness_stream;
  }

  [[nodiscard]] holoscan::expected<holomq::BufferHandle, holoscan::Error> export_handle()
      noexcept override {
    const std::lock_guard lock(slot_->mutex);
    auto handle = slot_->readiness_stream == nullptr
                      ? slot_->buffer.export_handle_unsynchronized()
                      : slot_->buffer.export_handle(slot_->readiness_stream);
    if (!handle) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "exporting persistent HoloMQ buffer failed: " + holomq::error_message(handle.error())});
    }
    return std::move(*handle);
  }

 private:
  std::shared_ptr<ExportSlot> slot_;
  std::atomic_uint32_t handle_count_{};
};

}  // namespace

struct ExportableHoloMQPool::Impl {
  Impl(holomq::Context context_value, holomq::BufferPool pool_value)
      : context(std::move(context_value)), pool(std::move(pool_value)) {}

  holomq::Context context;
  holomq::BufferPool pool;
  std::vector<std::shared_ptr<ExportSlot>> slots;
  std::size_t next_slot{};
  std::mutex acquire_mutex;
};

ExportableHoloMQPool::ExportableHoloMQPool(const std::size_t buffer_bytes,
                                           const std::uint32_t buffer_count,
                                           const std::int32_t cuda_device) {
  if (cudaSetDevice(cuda_device) != cudaSuccess) {
    throw std::runtime_error("selecting the CUDA device for the HoloMQ pool failed");
  }
  auto context = holomq::ContextBuilder{}.with_vmm_fd_export(true).create();
  if (!context) {
    throw std::runtime_error("creating the HoloMQ export context failed: " +
                             holomq::error_message(context.error()));
  }
  auto pool = context->create_device_buffer_pool(buffer_bytes, buffer_count);
  if (!pool) {
    throw std::runtime_error("creating the HoloMQ CUDA output pool failed: " +
                             holomq::error_message(pool.error()));
  }
  impl_ = std::make_unique<Impl>(std::move(*context), std::move(*pool));
  impl_->slots.reserve(buffer_count);
  for (std::uint32_t index = 0U; index < buffer_count; ++index) {
    auto buffer = impl_->pool.acquire();
    if (!buffer) {
      throw std::runtime_error("cold-binding HoloMQ CUDA output slot failed: " +
                               holomq::error_message(buffer.error()));
    }
    impl_->slots.push_back(std::make_shared<ExportSlot>(std::move(*buffer)));
  }
}

ExportableHoloMQPool::~ExportableHoloMQPool() = default;

holoscan::expected<holoscan::Tensor, holoscan::Error>
ExportableHoloMQPool::acquire_tensor(const holoscan::TensorShapeView shape,
                                     const DLDataType dtype) {
  std::shared_ptr<ExportSlot> selected;
  {
    const std::lock_guard pool_lock(impl_->acquire_mutex);
    for (std::size_t attempt = 0U; attempt < impl_->slots.size(); ++attempt) {
      const std::size_t index = (impl_->next_slot + attempt) % impl_->slots.size();
      const auto& slot = impl_->slots[index];
      bool expected = false;
      if (!slot->in_use.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      const std::lock_guard slot_lock(slot->mutex);
      if (!slot->first_generation && !slot->buffer.prepare_for_reuse()) {
        slot->in_use.store(false, std::memory_order_release);
        continue;
      }
      slot->first_generation = false;
      slot->readiness_stream = nullptr;
      impl_->next_slot = (index + 1U) % impl_->slots.size();
      selected = slot;
      break;
    }
  }
  if (!selected) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kBackpressured,
        "no persistent HoloMQ export slot has completed its imported generation"});
  }

  auto reference = std::make_shared<ExportableSlotReference>(selected);
  auto tensor = holoscan::make_tensor(shape,
                                      dtype,
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kDevice,
                                      selected->buffer.ptr(),
                                      std::move(reference),
                                      selected->buffer.cuda_device_ordinal().value_or(-1));
  if (!tensor) {
    selected->in_use.store(false, std::memory_order_release);
    return holoscan::make_unexpected(std::move(tensor).error());
  }
  return tensor;
}

}  // namespace holoscan::ops
