# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS v4l2-display-x11-base

COPY --chmod=755 holoscan_ridgerun /tmp/scripts/
RUN /tmp/scripts/holoscan_ridgerun env-info

RUN apt-get update \
    && apt-get install -y --no-install-recommends libx11-dev libx11-6 \
    && rm -rf /var/lib/apt/lists/*

FROM v4l2-display-x11-base AS v4l2-display-x11-live

FROM v4l2-display-x11-base AS v4l2-display-x11-validate
