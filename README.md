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
python torch2onnx.py
sh onnx2mlir.sh
sh mlir2tosa.sh
sh tosa2llvmir.sh
sh llvmir2out.sh
sh cpu_infer.sh
```

#### 注意点
* 编译整个Vit流程耗时非常长！！时间在1~2小时内，为了简化流程，这里使用的是Vit模型中的一个block。如果想要跑全模型，请将example/vit/torch2onnx.py的28、29行取消注释。第27行注释掉。

```python
#x = self.model.vit.encoder.layer[0](x)[0]
for layer in self.model.vit.encoder.layer:
   x = layer(x)[0]
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


#### 截至20230807:
* 目前进展：
1. 支持VIT所有算子的前端转换
2. 支持VIT所有算子下降到FP32
3. 支持VIT部分算子下降到NT8（不支持的用FP32）
4. 支持VIT最终下降到x86架构，FP32余弦相似度在0.99，INT8相似度在0.86

* 未完成的工作
1. top层前向推理（之前我觉得这个不重要，但是做量化的时候是用这个）
2. layergroup
3. INT8大算子量化，softmax，layernorm，conv算子量化
4. PTQ方式优化，VIT存在奇异值，目前方式是暴力法，阈值大于10的不量化
5. 下降到其他后端，例如nvidia arm
6. 写一个新的dialect，因为决赛就是新dialect
7. 全流程实现，目前是一张图测试，还需要前处理后处理以及输出结果

--------------------------------------------


#### TODO List:
* ~~xmake~~
* fastgelu
* layergroup
* 完全softmax、layernorm、conv等大算子的下降
* 完成tpu-mlir里面的run\_calibration功能
* 将weight-reorder在pass里面完成，而不是放到OnnxConvert中
* 下降到arm、nvdia后端
* 使用pdll来简化代码
* 重构代码，解决ConvPermute的问题
* 支持baby-llama
* 做类似profile的性能分析工具
