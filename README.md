# Tumor Generation in Longitudinal Mammograms via Transformer GAN with Adaptive Blending

PyTorch implementation of a projection-aware longitudinal tumor synthesis framework for mammograms. The model uses paired prior and current mammograms, view/side embeddings, temporal cross-attention, a variational latent unit, anatomically constrained blending, and a Swin-based discriminator.

## Repository Structure

```text
projection_aware_longitudinal_tumor_synthesis/
├── train.py
├── test.py
├── infer.py
├── configs/
├── data/
├── models/
├── losses/
├── engine/
├── metrics/
├── utils/
├── scripts/
└── assets/
```

## Expected Dataset Layout

```text
dataset/
├── train/
│   ├── metadata.csv
│   ├── prior/
│   ├── current/
│   ├── breast_masks/
│   └── tumor_masks/
├── val/
│   ├── metadata.csv
│   ├── prior/
│   ├── current/
│   ├── breast_masks/
│   └── tumor_masks/
└── test/
    ├── metadata.csv
    ├── prior/
    ├── current/
    ├── breast_masks/
    └── tumor_masks/
```

### `metadata.csv`

```csv
case_id,prior_path,current_path,breast_mask_path,tumor_mask_path,view,side,label
0001,train/prior/0001.png,train/current/0001.png,train/breast_masks/0001.png,train/tumor_masks/0001.png,CC,Left,1
0002,train/prior/0002.png,train/current/0002.png,train/breast_masks/0002.png,,MLO,Right,0
```

- `label=1` indicates cancer case.
- `tumor_mask_path` can be empty for normal cases.
- `view` should be one of `CC`, `MLO`.
- `side` should be one of `Left`, `Right`.

## Installation

```bash
conda create -n proj_tumor python=3.10 -y
conda activate proj_tumor
pip install -r requirements.txt
```

## Training

```bash
python train.py \
  --data_root /path/to/dataset \
  --train_csv train/metadata.csv \
  --val_csv val/metadata.csv \
  --output_dir runs/exp1
```

## Testing

```bash
python test.py \
  --data_root /path/to/dataset \
  --test_csv test/metadata.csv \
  --checkpoint /path/to/checkpoints/best_generator.pt
```

## Inference

```bash
python infer.py \
  --data_root /path/to/dataset \
  --csv_path test/metadata.csv \
  --checkpoint /path/to/checkpoints/best_generator.pt \
  --save_dir outputs/inference
```

## GitHub Commands

```bash
git init
git add .
git commit -m "Initial commit: projection-aware longitudinal tumor synthesis"
git branch -M main
git remote add origin https://github.com/<username>/<repo_name>.git
git push -u origin main
```
