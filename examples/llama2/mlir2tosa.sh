mini-opt --init --convert-top-to-tosa="includeWeight=True weightType=FP32"  --tosa-op-fold --deinit llama2_split.mlir  -o tosa.mlir

sed -i 's/main/entry/g' tosa.mlir
