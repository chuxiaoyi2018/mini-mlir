convert onnx model to top mlir file

### Step0
clone mini-mlir
```
git clone git@github.com:chuxiaoyi2018/mini-mlir.git
```

### Step1
build llvm(oneDNN/cnpy)
```
cd onnx2top
sh build_third_party.sh
```

run these command to check

if you cannot "import mlir"

you may run step2 ahead
```
python3
>>> import mlir 
```

### Step2
source environment
```
source mini_envsetup.sh
```

run these command to check
```
model_transform.py --help
```

### Step3
convert onnx model to top mlir
```
cd examples
python relu.py
sh onnx2mlir.sh
```

if the terminal prints:
```
Save mlir file: model/relu.mlir
```
Success!
