# mini-mlir


### how to compile the project
```
git clone https://github.com/chuxiaoyi2018/mini-mlir.git --depth 1
source mini_envsetup.sh
./build.sh
```

### how to run examples
```
cd example/vit
python torch2onnx.sh
sh onnx2mlir.sh
sh top2tosa.sh
sh tosa2llvmir.sh
sh llvmir2out.sh
sh cpu_infer.sh
```


---------------------------------------------------------------------
### 版本变更

#### 第二版

【版本改动】
* 将所有tops都改为了top
* 增加了top到tosa的conversion
* 增加了bindings，可以支持top的推理，在python/tools中，为mlir_inference
* top层增加了几个算子


#### 第三版

可以通过下载这个来获取release包

https://github.com/chuxiaoyi2018/mini-mlir/releases/tag/mini-mlir

【版本改动】
* 目前可以执行从onnx->top->tosa->llvmir->final.o
* 添加了third_party，直接用tpu-mlir的llvm、oneDNN、cnpy


#### 第四版


【版本改动】
* 增加了很多FP32 tosa算子，目前可以从将VIT模型下降到FP32，并且相似度在0.99
* 增加了INT8 tosa算子，目前完成的是permute、reshape、matmul、gelu等，可以将VIT先下降到INT8，再将无法下降的下降到FP32
* 增加了很多针对INT8图优化的算子，例如DoubleCast、DoubleRecipocal

#### 第五版


【版本改动】
* 做了一个代码的重构，使得目录更为清晰简洁

--------------------------------------------




#### TODO List:
* xmake
* fastgelu
* layergroup
* 完全softmax、layernorm、conv等大算子的下降
* 完成tpu-mlir里面的run\_calibration功能
* 将weight-reorder在pass里面完成，而不是放到OnnxConvert中
* 下降到arm、nvdia后端
