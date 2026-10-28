# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

add_library(v4l2_userptr_source v4l2_userptr_source.cpp)
add_library(holoscan::v4l2_userptr_source ALIAS v4l2_userptr_source)

target_compile_features(v4l2_userptr_source PUBLIC cxx_std_20)
target_include_directories(v4l2_userptr_source
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(v4l2_userptr_source
    PUBLIC holoscan::operator_utils holoscan::flow_trace
    PRIVATE CUDA::cudart
)
