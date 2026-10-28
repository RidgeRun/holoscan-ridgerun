// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <xf86drmMode.h>

namespace holoscan::ops {

/**
 * @brief Owns a DRM/KMS device, its selected connector/CRTC, and the saved CRTC state.
 *
 * The class deliberately does not own framebuffers. Callers must restore the CRTC before
 * destroying the buffers that are currently attached to it.
 */
class DrmDevice final {
 public:
  DrmDevice(int width,
            int height,
            std::string device,
            std::uint32_t requested_connector,
            std::string requested_mode);
  ~DrmDevice();

  DrmDevice(const DrmDevice&) = delete;
  DrmDevice& operator=(const DrmDevice&) = delete;

  void start();
  void set_framebuffer(std::uint32_t framebuffer);
  void restore_crtc() noexcept;
  void close() noexcept;

  [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }
  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] std::uint32_t connector_id() const noexcept { return connector_id_; }
  [[nodiscard]] std::uint32_t crtc_id() const noexcept { return crtc_id_; }
  [[nodiscard]] const drmModeModeInfo& mode() const noexcept { return mode_; }
  [[nodiscard]] const std::string& path() const noexcept { return device_; }

 private:
  [[nodiscard]] std::optional<drmModeModeInfo> choose_mode(
      const drmModeConnector& connector) const;
  [[nodiscard]] static std::uint32_t choose_crtc(const drmModeRes& resources,
                                                 const drmModeEncoder& encoder);

  int width_{};
  int height_{};
  std::string device_;
  std::uint32_t requested_connector_{};
  std::string requested_mode_;
  drmModeModeInfo mode_{};
  drmModeCrtc* saved_crtc_{};
  std::uint32_t connector_id_{};
  std::uint32_t crtc_id_{};
  int fd_{-1};
  bool drm_master_{};
  bool crtc_restored_{};
};

}  // namespace holoscan::ops
