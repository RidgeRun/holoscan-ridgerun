# YOLO11 inference for IMX274

This Holoscan SDK (HSDK) 5 C++ application captures IMX274 RG10 frames through
V4L2 USERPTR, prepares a YOLO11 input tensor on CUDA, runs TensorRT inference,
and consumes the output with `FakesinkOp`. It is intentionally headless and it
does not decode detections, draw bounding boxes, or display frames.

## Pipeline

```text
V4L2UserPtrSourceOp
  -> RawBayerToYoloTensorOp
  -> TensorRtInferenceOp
  -> FakesinkOp
```

The fused CUDA preprocessor converts RG10 Bayer data directly into a normalized
RGB NCHW tensor with shape `[1,3,640,640]`. It applies aspect-ratio-preserving
resize and centered letterbox padding without creating a full-resolution BGRA
image. The pipeline does not perform display enhancement, result
postprocessing, NMS, overlays, compositing, or device-to-host result copies.

## Model compatibility

Select the model with `--model`. The argument may point to a serialized
TensorRT engine or an ONNX model. Replace the relative paths in the commands
below with the location of your own model:

```text
models/yolo11.engine
models/yolo11.onnx
```

Relative paths are resolved from the application's working directory. Ensure
the model is available at that location, or adjust the path passed through
`--run-args`. When running in a container, mount or copy the model into the
container at the corresponding path.

The current inference path expects:

- one static float32 input compatible with `[1,3,640,640]`;
- one static float32 output no larger than 16 MiB;
- preprocessing compatible with normalized RGB NCHW and centered letterboxing.

The headless application does not interpret the output tensor. It therefore
does not require a specific YOLO11 output layout such as decoded
`[1,8400,6]`; an undecoded layout such as `[1,84,8400]` can also be consumed
by `FakesinkOp`. A model that needs different input dimensions, tensor types,
multiple inputs or outputs, or different normalization requires corresponding
operator changes.

TensorRT engines are tied to the target GPU and TensorRT version. If an engine
cannot be deserialized, provide the ONNX model and an optional relative cache
path:

```bash
./holoscan_ridgerun run yolo11_userptr_inference live \
  --local --language cpp --no-local-build \
  --run-args='--model models/yolo11.onnx --engine-cache models/yolo11_hsdk.engine --frame-limit 1'
```

Replace `models/yolo11.onnx` and `models/yolo11_hsdk.engine` with the desired
ONNX and generated-engine locations.

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build yolo11_userptr_inference live \
  --local --language cpp
./holoscan_ridgerun run yolo11_userptr_inference validate \
  --local --language cpp --no-local-build
```

Validation compiles the graph without opening the camera or loading the model.

## Run

The following command uses a relative placeholder model path. Replace
`models/yolo11.engine` with the path to the TensorRT engine or ONNX model that
you want to execute:

```bash
./holoscan_ridgerun run yolo11_userptr_inference live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --model models/yolo11.engine --width 1920 --height 1080 --fps 60 --frame-limit 500'
```

`FakesinkOp` discards every inference output on CUDA. Press Ctrl-C to stop an
unlimited run.

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
