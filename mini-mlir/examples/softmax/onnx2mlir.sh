model_transform.py \
    --model_name softmax \
    --model_def softmax.onnx \
    --input_shapes [[1,2,2,72]] \
    --mlir softmax.mlir \
    --model_type onnx
