# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(operator_utils INTERFACE)
add_library(holoscan::operator_utils ALIAS operator_utils)

target_compile_features(operator_utils INTERFACE cxx_std_20)
target_include_directories(operator_utils
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(operator_utils INTERFACE holoscan::core)
