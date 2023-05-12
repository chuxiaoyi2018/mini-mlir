model_transform.py \
    --model_name gemm \
    --model_def gemm.pt \
    --input_shapes [[2,72],[72,1]] \
    --mlir gemm.mlir 
