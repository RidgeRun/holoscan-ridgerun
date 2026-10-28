# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS argus-camera-x11-base

COPY --chmod=755 holoscan_ridgerun /tmp/scripts/
RUN /tmp/scripts/holoscan_ridgerun env-info

# The Holoscan CUDA 13 image supplies CUDA and Holoscan, but not Jetson's
# Multimedia API headers. Extract only the JP 7.2.1 (L4T r39.2) Multimedia API
# and camera runtime packages; installing nvidia-jetpack-dev in this image
# would pull a second, incompatible CUDA stack.
RUN dpkg --print-architecture | grep -qx arm64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgles2-mesa-dev \
        libx11-dev \
    && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
        -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
    && printf '%s\n' \
        'deb https://repo.download.nvidia.com/jetson/common r39.2 main' \
        'deb https://repo.download.nvidia.com/jetson/som r39.2 main' \
        > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && apt-get update \
    && apt-get download \
        nvidia-l4t-camera=39.2.1-20260806224157 \
        nvidia-l4t-jetson-multimedia-api=39.2.1-20260806224157 \
    && mkdir -p /tmp/argus-camera /tmp/jetson-multimedia-api \
    && dpkg-deb -x nvidia-l4t-camera_*_arm64.deb /tmp/argus-camera \
    && dpkg-deb -x nvidia-l4t-jetson-multimedia-api_*_arm64.deb /tmp/jetson-multimedia-api \
    && mkdir -p /usr/lib/aarch64-linux-gnu/nvidia \
    && cp -a /tmp/argus-camera/usr/lib/aarch64-linux-gnu/nvidia/. \
        /usr/lib/aarch64-linux-gnu/nvidia/ \
    && cp -a /tmp/jetson-multimedia-api/usr/src/jetson_multimedia_api \
        /usr/src/jetson_multimedia_api \
    && ldconfig \
    && rm -rf \
        /tmp/argus-camera \
        /tmp/jetson-multimedia-api \
        nvidia-l4t-camera_*_arm64.deb \
        nvidia-l4t-jetson-multimedia-api_*_arm64.deb \
        /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

FROM argus-camera-x11-base AS argus-camera-x11-live

# The validation target uses the same compile-time dependencies while avoiding
# camera and display initialization at application runtime.
FROM argus-camera-x11-base AS argus-camera-x11-validate
