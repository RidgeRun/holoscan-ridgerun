# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(v4l2_display v4l2_display.cu)
target_compile_features(v4l2_display PRIVATE cxx_std_20)
set_target_properties(v4l2_display
    PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
)
target_compile_options(v4l2_display
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
)
target_link_libraries(v4l2_display
    PRIVATE
        holoscan::drm_bgra_display
        holoscan::drm_display
        holoscan::raw_bayer_to_bgra
        holoscan::v4l2_source
        holoscan::v4l2_userptr_source
        holoscan::core
        Threads::Threads
)
