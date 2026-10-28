# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "argus_camera is only supported on Jetson aarch64")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)

find_path(ARGUS_INCLUDE_DIR
    NAMES Argus/Argus.h
    HINTS /usr/src/jetson_multimedia_api/argus/include
    DOC "Jetson Multimedia API Argus include directory"
)
if(NOT ARGUS_INCLUDE_DIR)
    message(FATAL_ERROR
        "Argus headers were not found. Install the JetPack development components "
        "so /usr/src/jetson_multimedia_api/argus/include/Argus/Argus.h exists."
    )
endif()

find_library(ARGUS_SOCKET_CLIENT_LIBRARY
    NAMES nvargus_socketclient
    HINTS /usr/lib/aarch64-linux-gnu/nvidia
    REQUIRED
)

add_library(argus_camera argus_camera.cu)
add_library(holoscan::argus_camera ALIAS argus_camera)

target_compile_features(argus_camera PUBLIC cxx_std_20)
set_target_properties(argus_camera PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(argus_camera
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
)
target_include_directories(argus_camera
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        ${ARGUS_INCLUDE_DIR}
)
target_link_libraries(argus_camera
    PUBLIC
        holoscan::operator_utils holoscan::flow_trace
        CUDA::cuda_driver
        CUDA::cudart
        OpenGL::EGL
    PRIVATE
        ${ARGUS_SOCKET_CLIENT_LIBRARY}
)
