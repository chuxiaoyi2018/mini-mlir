# mini-mlir

#### 第二版

【版本改动】
* 将所有tops都改为了top
* 增加了top到tosa的conversion
* 增加了bindings，可以支持top的推理，在python/tools中，为mlir_inference
* top层增加了几个算子


#### 第三版

【版本改动】
* 目前可以执行从onnx->top->tosa->llvmir->final.o
* 添加了third_party，直接用tpu-mlir的llvm、oneDNN、cnpy
