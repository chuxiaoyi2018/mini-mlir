model_transform.py \
    --model_name relu \
    --model_def model/relu.onnx \
    --input_shapes [[2,72]] \
    --mlir model/relu.mlir \
    --model_type onnx
