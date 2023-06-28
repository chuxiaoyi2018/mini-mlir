model_transform.py \
    --model_name matmul \
    --model_def matmul.onnx \
    --input_shapes [2,3] \
    --mlir matmul.mlir \
    --model_type onnx
