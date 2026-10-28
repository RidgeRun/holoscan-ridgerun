# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(yolo11_userptr_inference
    yolo11_inference_headless.cu
    raw_bayer_to_yolo_tensor.cu
)
target_compile_features(yolo11_userptr_inference PRIVATE cxx_std_20)
set_target_properties(yolo11_userptr_inference PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)
target_compile_options(yolo11_userptr_inference
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
)
target_include_directories(yolo11_userptr_inference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(yolo11_userptr_inference
    PRIVATE
        holoscan::fakesink_op
        holoscan::tensorrt_inference
        holoscan::v4l2_userptr_source
        holoscan::core
        CUDA::cudart
        Threads::Threads
)
