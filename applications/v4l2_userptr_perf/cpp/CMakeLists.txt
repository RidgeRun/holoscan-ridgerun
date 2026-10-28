# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(v4l2_userptr_perf v4l2_userptr_perf.cpp)
target_compile_features(v4l2_userptr_perf PRIVATE cxx_std_20)
target_link_libraries(v4l2_userptr_perf
    PRIVATE
        holoscan::perf_op
        holoscan::raw_bayer_to_bgra
        holoscan::v4l2_userptr_source
        holoscan::x11_display
        holoscan::core
        Threads::Threads
)
