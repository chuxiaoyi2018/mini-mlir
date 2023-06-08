# mini-mlir

#### 第二版的docker
http://disk-sophgo-vip.quickconnect.cn/sharing/cMgVR8Ssu

【版本改动】
* 将所有tops都改为了top
* 增加了top到tosa的conversion
* 增加了bindings，可以支持top的推理，在python/tools中，为mlir_inference
* top层增加了几个算子


#### 第三版的改动（第二十九次提交）
第三版的代码http://disk-sophgo-vip.quickconnect.cn/sharing/C9jGgLDlg ，包含third_part和编译文件，比较大4.8G

【版本改动】
* 完成了top2tosa的工作，可以生成tosa.mlir文件
* 添加了top2tosa的add算子，在mini-mlir/examples/add的top2tosa.sh下
