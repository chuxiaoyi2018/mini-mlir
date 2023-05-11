This folder largely borrows from [this repo](https://github.com/megvii-research/FQ-ViT). While the original repo focuses on the full quantization of ViT (as seen in vit_quant.py), this folder focuses on the usage and modification of different quantization components (as seen in subfolder models/ptq).

## Getting Started

### Environment

- torch 2.0
- cuda 11.7

### Examples

All usage examples are in tests folder.

### Run

- Step 1: Set up the environment varaibles

```bash
source envsetup.sh
```

- Step 2: Run the test

```bash
python test_toy.py toy
```

```bash
python test_toy.py toy --quant
```
