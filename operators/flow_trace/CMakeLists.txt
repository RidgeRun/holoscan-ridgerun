# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(flow_trace INTERFACE)
add_library(holoscan::flow_trace ALIAS flow_trace)
target_compile_features(flow_trace INTERFACE cxx_std_20)
target_include_directories(flow_trace
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(flow_trace INTERFACE holoscan::core)
