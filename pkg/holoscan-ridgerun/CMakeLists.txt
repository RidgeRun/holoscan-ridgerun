# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is entered via add_holohub_package(holoscan-ridgerun ...) in
# pkg/CMakeLists.txt, which only fires when PKG_holoscan_ridgerun is ON.
include(holohub_configure_deb)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(HOLOSCAN_DEB_DEPENDENCY "holoscan (>= 5.0.0)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(HOLOSCAN_DEB_DEPENDENCY "holoscan-cuda-13 (>= 5.0.0)")
else()
  message(FATAL_ERROR "Unsupported Debian package architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

holohub_configure_deb(
    NAME        "holoscan-ridgerun"
    DESCRIPTION "${PROJECT_DESCRIPTION}"
    VERSION     "${PROJECT_VERSION}"
    VENDOR      "RidgeRun, LLC"
    CONTACT     "RidgeRun"
    # Match the platform-specific SDK package for the CUDA 13 base image.
    DEPENDS     "${HOLOSCAN_DEB_DEPENDENCY}"
    EXPORT_NAME "holoscan_ridgerun_targets"
)
