# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

find_path(LIBDRM_INCLUDE_DIR xf86drm.h PATH_SUFFIXES libdrm REQUIRED)
find_path(LIBDRM_DRM_INCLUDE_DIR drm.h PATH_SUFFIXES libdrm REQUIRED)
find_library(LIBDRM_LIBRARY drm REQUIRED)

add_library(drm_display drm_display.cpp drm_device.cpp drm_buffer.cpp drm_display.cu)
add_library(holoscan::drm_display ALIAS drm_display)

target_compile_features(drm_display PUBLIC cxx_std_20)
target_include_directories(drm_display
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    PRIVATE
        ${LIBDRM_INCLUDE_DIR}
        ${LIBDRM_DRM_INCLUDE_DIR}
)
target_link_libraries(drm_display
    PUBLIC
        holoscan::operator_utils holoscan::flow_trace
    PRIVATE
        ${LIBDRM_LIBRARY}
        CUDA::cuda_driver
        CUDA::cudart
)
