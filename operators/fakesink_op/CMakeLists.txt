# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(fakesink_op fakesink_op.cpp)
add_library(holoscan::fakesink_op ALIAS fakesink_op)

target_compile_features(fakesink_op PUBLIC cxx_std_20)
target_include_directories(fakesink_op
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(fakesink_op PUBLIC holoscan::core holoscan::flow_trace)
