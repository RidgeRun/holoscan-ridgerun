# Flow trace

`FlowTraceOp` forwards complete Tensor samples without accessing or copying
their payload and records a named timestamp interval for every frame. The
recorder writes the text and cross-process event formats consumed by the
project benchmarking tools. It is independent of camera, display, and
image-format operators and is enabled only when
`HOLOSCAN_FLOW_TRACKING_LOG_FILE` is set.
