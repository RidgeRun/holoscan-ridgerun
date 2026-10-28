# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(test_pattern test_pattern.cu)
add_library(holoscan::test_pattern ALIAS test_pattern)

target_compile_features(test_pattern PUBLIC cxx_std_20)
set_target_properties(test_pattern PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(test_pattern
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
)
target_include_directories(test_pattern
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(test_pattern
    PUBLIC holoscan::core
    PRIVATE CUDA::cudart
)
