# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(interprocess_src
    interprocess_reader.cpp
    interprocess_src.cpp
)
add_library(holoscan::interprocess_src ALIAS interprocess_src)

target_compile_features(interprocess_src PUBLIC cxx_std_20)
target_include_directories(interprocess_src
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(interprocess_src
    PUBLIC
        holoscan::core
        holoscan::operator_utils
        holoscan::flow_trace
)
