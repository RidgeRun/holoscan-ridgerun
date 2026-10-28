# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS yolo11-userptr-inference-base

COPY --chmod=755 holoscan_ridgerun /tmp/scripts/
RUN /tmp/scripts/holoscan_ridgerun env-info

# The serialized engine is supplied at runtime because TensorRT engines are
# tied to the target GPU and TensorRT stack.
FROM yolo11-userptr-inference-base AS yolo11-userptr-inference-live

FROM yolo11-userptr-inference-base AS yolo11-userptr-inference-validate
