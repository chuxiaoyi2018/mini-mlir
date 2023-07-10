mini-opt --init --convert-top-to-tosa="includeWeight=True" --deinit vit.mlir  -o tosa.mlir

sed -i 's/main/entry/g' tosa.mlir
