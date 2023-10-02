mini-opt --init --convert-top-to-tosa="includeWeight=True weightType=INT8 tableFile=vit_cali_table.csv"  --tosa-op-fold --deinit vit.mlir  -o tosa.mlir

sed -i 's/main/entry/g' tosa.mlir
