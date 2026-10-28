# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(cpp)
