model_transform.py \
    --model_name llama2 \
    --model_def llama2-lite.onnx \
    --input_shapes [[1,256]] \
    --mlir llama2.mlir \
    --model_type onnx \
    --chip cpu
