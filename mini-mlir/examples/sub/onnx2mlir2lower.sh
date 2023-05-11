op_name="sub"

model_transform.py \
    --model_name $op_name \
    --model_def $op_name.onnx \
    --input_shapes [[2,72],[2,72]] \
    --mlir $op_name.mlir \
    --model_type onnx


mini-opt  $op_name.mlir -o ${op_name}_lower.mlir
