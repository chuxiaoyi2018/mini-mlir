model_transform.py \
    --model_name vit \
    --model_def vit-lite.onnx* \
    --input_shapes [[1,224,224,3]] \
    --mlir vit.mlir \
    --model_type onnx \
    --chip cpu
