// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace holoscan::ops {

/// Copyable shared gate for sources that must drain input before publication starts.
class FrameStartGate final {
 public:
  explicit FrameStartGate(bool open = true)
      : open_(std::make_shared<std::atomic_bool>(open)) {}

  void open() noexcept { open_->store(true, std::memory_order_release); }
  [[nodiscard]] bool is_open() const noexcept {
    return open_->load(std::memory_order_acquire);
  }

 private:
  std::shared_ptr<std::atomic_bool> open_;
};

class FrameLimit final {
 public:
  explicit FrameLimit(std::uint64_t limit = 0U)
      : limit_(limit), frames_(std::make_shared<std::atomic<std::uint64_t>>()) {}

  [[nodiscard]] bool reached() const noexcept {
    return limit_ != 0U && frames_->load(std::memory_order_acquire) >= limit_;
  }

  void record_frame() noexcept {
    frames_->fetch_add(1U, std::memory_order_release);
  }

 private:
  std::uint64_t limit_;
  std::shared_ptr<std::atomic<std::uint64_t>> frames_;
};

}  // namespace holoscan::ops
