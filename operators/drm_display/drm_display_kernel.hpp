// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::ops {

/**
 * Convert packed YUYV into the BGRA layout expected by the DRM XRGB8888
 * framebuffer. The destination is an externally imported DRM DMABUF mapped
 * into CUDA; no intermediate CUDA allocation is made.
 */
cudaError_t launch_yuyv_to_bgra(const std::uint8_t* input,
                                std::uint8_t* output,
                                std::size_t output_pitch,
                                std::int32_t width,
                                std::int32_t height,
                                cudaStream_t stream) noexcept;

}  // namespace holoscan::ops
