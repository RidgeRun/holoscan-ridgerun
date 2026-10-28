// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "drm_display/drm_device.hpp"

#include <drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <xf86drm.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace holoscan::ops {
namespace {

[[nodiscard]] std::string errno_message(std::string_view operation) {
  std::ostringstream message;
  message << operation << " failed: " << std::strerror(errno);
  return message.str();
}

[[nodiscard]] std::string mode_name(const drmModeModeInfo& mode) {
  std::ostringstream name;
  name << mode.hdisplay << 'x' << mode.vdisplay;
  if (mode.vrefresh > 0) {
    name << '@' << mode.vrefresh;
  }
  return name.str();
}

[[nodiscard]] std::optional<std::pair<int, int>> parse_dimensions(std::string_view text) {
  const std::size_t separator = text.find('x');
  if (separator == std::string_view::npos || separator == 0U || separator + 1U >= text.size()) {
    return std::nullopt;
  }
  try {
    const int width = std::stoi(std::string{text.substr(0U, separator)});
    const int height = std::stoi(std::string{text.substr(separator + 1U)});
    if (width <= 0 || height <= 0) {
      return std::nullopt;
    }
    return std::pair{width, height};
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace

DrmDevice::DrmDevice(int width,
                     int height,
                     std::string device,
                     std::uint32_t requested_connector,
                     std::string requested_mode)
    : width_(width),
      height_(height),
      device_(std::move(device)),
      requested_connector_(requested_connector),
      requested_mode_(std::move(requested_mode)) {}

DrmDevice::~DrmDevice() { close(); }

void DrmDevice::start() {
  close();
  try {
    fd_ = ::open(device_.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
      throw std::runtime_error(errno_message("open(" + device_ + ")"));
    }

    std::uint64_t capability = 0U;
    if (drmGetCap(fd_, DRM_CAP_DUMB_BUFFER, &capability) != 0 || capability == 0U) {
      throw std::runtime_error("DRM device does not support dumb buffers");
    }
    if (drmGetCap(fd_, DRM_CAP_PRIME, &capability) != 0 ||
        (capability & DRM_PRIME_CAP_EXPORT) == 0U) {
      throw std::runtime_error("DRM device cannot export framebuffer DMABUFs");
    }
    if (drmSetMaster(fd_) != 0) {
      throw std::runtime_error(
          errno_message("drmSetMaster; stop the display manager before using DRM exclusive mode"));
    }
    drm_master_ = true;

    drmModeRes* resources = drmModeGetResources(fd_);
    if (resources == nullptr) {
      throw std::runtime_error(errno_message("drmModeGetResources"));
    }

    drmModeConnector* connector = nullptr;
    if (requested_connector_ != 0U) {
      connector = drmModeGetConnector(fd_, requested_connector_);
    } else {
      for (int index = 0; index < resources->count_connectors; ++index) {
        drmModeConnector* candidate = drmModeGetConnector(fd_, resources->connectors[index]);
        if (candidate == nullptr) {
          continue;
        }
        if (candidate->connection == DRM_MODE_CONNECTED && candidate->count_modes > 0) {
          connector = candidate;
          break;
        }
        drmModeFreeConnector(candidate);
      }
    }

    if (connector == nullptr || connector->connection != DRM_MODE_CONNECTED ||
        connector->count_modes == 0) {
      if (connector != nullptr) {
        drmModeFreeConnector(connector);
      }
      drmModeFreeResources(resources);
      throw std::runtime_error("no connected DRM connector with modes was found");
    }
    connector_id_ = connector->connector_id;

    const auto selected_mode = choose_mode(*connector);
    if (!selected_mode.has_value()) {
      std::ostringstream message;
      message << "connector " << connector_id_ << " has no mode matching " << width_ << 'x'
              << height_ << "; available modes:";
      for (int index = 0; index < connector->count_modes; ++index) {
        message << ' ' << mode_name(connector->modes[index]);
      }
      drmModeFreeConnector(connector);
      drmModeFreeResources(resources);
      throw std::runtime_error(message.str());
    }
    mode_ = *selected_mode;

    drmModeEncoder* encoder = nullptr;
    if (connector->encoder_id != 0U) {
      encoder = drmModeGetEncoder(fd_, connector->encoder_id);
    }
    if (encoder == nullptr) {
      for (int index = 0; index < connector->count_encoders; ++index) {
        encoder = drmModeGetEncoder(fd_, connector->encoders[index]);
        if (encoder != nullptr) {
          break;
        }
      }
    }
    if (encoder == nullptr) {
      drmModeFreeConnector(connector);
      drmModeFreeResources(resources);
      throw std::runtime_error("could not find an encoder for the DRM connector");
    }

    crtc_id_ = choose_crtc(*resources, *encoder);
    drmModeFreeEncoder(encoder);
    if (crtc_id_ == 0U) {
      drmModeFreeConnector(connector);
      drmModeFreeResources(resources);
      throw std::runtime_error("could not find a compatible DRM CRTC");
    }

    saved_crtc_ = drmModeGetCrtc(fd_, crtc_id_);
    if (saved_crtc_ == nullptr) {
      drmModeFreeConnector(connector);
      drmModeFreeResources(resources);
      throw std::runtime_error(errno_message("drmModeGetCrtc"));
    }
    crtc_restored_ = false;

    drmModeFreeConnector(connector);
    drmModeFreeResources(resources);
  } catch (const std::exception& exception) {
    close();
    throw std::runtime_error(std::string{"DRM start failed: "} + exception.what());
  }
}

void DrmDevice::set_framebuffer(std::uint32_t framebuffer) {
  std::uint32_t connector_ids[] = {connector_id_};
  if (drmModeSetCrtc(fd_,
                     crtc_id_,
                     framebuffer,
                     0,
                     0,
                     connector_ids,
                     1,
                     &mode_) != 0) {
    throw std::runtime_error(errno_message("drmModeSetCrtc"));
  }
  crtc_restored_ = false;
}

void DrmDevice::restore_crtc() noexcept {
  if (fd_ < 0 || !drm_master_ || saved_crtc_ == nullptr || crtc_restored_) {
    return;
  }
  std::uint32_t connector_ids[] = {connector_id_};
  if (drmModeSetCrtc(fd_,
                     saved_crtc_->crtc_id,
                     saved_crtc_->buffer_id,
                     saved_crtc_->x,
                     saved_crtc_->y,
                     connector_ids,
                     1,
                     saved_crtc_->mode_valid ? &saved_crtc_->mode : nullptr) == 0) {
    crtc_restored_ = true;
  }
}

void DrmDevice::close() noexcept {
  restore_crtc();
  if (saved_crtc_ != nullptr) {
    drmModeFreeCrtc(saved_crtc_);
    saved_crtc_ = nullptr;
  }
  if (drm_master_ && fd_ >= 0) {
    static_cast<void>(drmDropMaster(fd_));
    drm_master_ = false;
  }
  if (fd_ >= 0) {
    static_cast<void>(::close(fd_));
    fd_ = -1;
  }
  connector_id_ = 0U;
  crtc_id_ = 0U;
  crtc_restored_ = false;
}

std::optional<drmModeModeInfo> DrmDevice::choose_mode(
    const drmModeConnector& connector) const {
  std::pair<int, int> dimensions{width_, height_};
  if (!requested_mode_.empty()) {
    const auto parsed = parse_dimensions(requested_mode_);
    if (!parsed.has_value() || parsed->first != width_ || parsed->second != height_) {
      return std::nullopt;
    }
    dimensions = *parsed;
  }

  const drmModeModeInfo* best = nullptr;
  for (int index = 0; index < connector.count_modes; ++index) {
    const drmModeModeInfo& candidate = connector.modes[index];
    if (candidate.hdisplay != dimensions.first || candidate.vdisplay != dimensions.second) {
      continue;
    }
    if (best == nullptr ||
        ((candidate.type & DRM_MODE_TYPE_PREFERRED) != 0 &&
         (best->type & DRM_MODE_TYPE_PREFERRED) == 0) ||
        candidate.vrefresh > best->vrefresh) {
      best = &candidate;
    }
  }
  return best == nullptr ? std::nullopt : std::optional<drmModeModeInfo>{*best};
}

std::uint32_t DrmDevice::choose_crtc(const drmModeRes& resources,
                                     const drmModeEncoder& encoder) {
  if (encoder.crtc_id != 0U) {
    for (int index = 0; index < resources.count_crtcs; ++index) {
      if (resources.crtcs[index] == encoder.crtc_id) {
        return encoder.crtc_id;
      }
    }
  }
  for (int index = 0; index < resources.count_crtcs; ++index) {
    if ((encoder.possible_crtcs & (1U << index)) != 0U) {
      return resources.crtcs[index];
    }
  }
  return 0U;
}

}  // namespace holoscan::ops
