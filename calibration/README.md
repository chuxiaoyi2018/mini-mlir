# Calibration


- Setp 1

```bash
source envsetup.sh
```

- Step 2

```bash
run_calibration.py <path-to-the-mlir-model> \
    --dataset <path-to-the-calibration-set> \
    --input_num <num> \
    -o <output-calibration-table-file-name>
```

e.g.

```bash
run_calibration.py model/vit-base-patch16-224.mlir \
    --dataset caliset \
    --input_num 20 \
    -o vit_cali_table
```
