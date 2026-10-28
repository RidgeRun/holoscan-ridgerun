# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(raw_bayer_to_bgra
    raw_bayer_to_bgra.cu
    exportable_holomq_pool.cpp
)
add_library(holoscan::raw_bayer_to_bgra ALIAS raw_bayer_to_bgra)

target_compile_features(raw_bayer_to_bgra PUBLIC cxx_std_20)
set_target_properties(raw_bayer_to_bgra PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(raw_bayer_to_bgra
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(raw_bayer_to_bgra
    PUBLIC holoscan::operator_utils holoscan::flow_trace
    PRIVATE CUDA::cudart holoscan::holomq
)
