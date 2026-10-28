# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(perf_op perf_op.cpp)
add_library(holoscan::perf_op ALIAS perf_op)

target_compile_features(perf_op PUBLIC cxx_std_20)
target_include_directories(perf_op
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(perf_op PUBLIC holoscan::core Threads::Threads)
