The README is on the way.

<!-- ## Getting Started

### Environment

- Install PyTorch and torchvision. e.g.,

```bash
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch
```

### Run


Example: Evaluate quantized vit_base with MinMax quantizer and PTF and LIS.

```bash
python test_quant.py vit_base <YOUR_DATA_DIR> --quant --ptf --lis --quant-method minmax
```

```bash
python test_quant.py vit_base /home/hujunhao/hdd-hujunhao/tiny-imagenet-200 --quant --quant-method minmax
```

- `vit_small`: model architecture, which can be replaced by `vit_base`, `vit_large`.

- `--quant`: whether to quantize the model.

- `--ptf`: whether to use **Power-of-Two Factor Integer Layernorm**.

- `--lis`: whether to use **Log-Integer-Softmax**.

- `--quant-method`: quantization methods of activations, which can be chosen from `minmax`, `ema`, `percentile` and `omse`. -->
