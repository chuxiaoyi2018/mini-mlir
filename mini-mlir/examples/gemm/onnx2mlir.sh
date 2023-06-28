model_transform.py \
    --model_name gemm \
    --model_def gemm.onnx \
    --input_shapes [2,3] \
    --mlir gemm.mlir \
    --model_type onnx
