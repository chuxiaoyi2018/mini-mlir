# mini-mlir

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


how to compile the project
```
cd /workspace
git clone https://github.com/chuxiaoyi2018/mini-mlir.git --depth 1
cd mini-mlir/mini-mlir
source mini_envsetup.sh
./build.sh
```

run examples
```
cd mini-mlir/examples/add
python3 add.py
sh onnx2mlir.sh
sh top2tosa.sh
sh tosa2llvmir.sh
sh llvmir2out.sh
```
