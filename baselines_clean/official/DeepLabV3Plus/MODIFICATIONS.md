# MODIFICATIONS.md -- DeepLabV3+ (deeplab)

## Source

| Field | Value |
|-------|-------|
| Type | External baseline |
| Paper | "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (Chen et al., ECCV 2018) |
| Official GitHub URL | https://github.com/VainF/DeepLabV3Plus-Pytorch |
| Commit | 4e1087de |
| Clone date | 2026-05-02 |
| Dirty (uncommitted changes in repo/) | 0 |

## Table name

DeepLabV3+ (ResNet-101)

## Purpose

Representative segmentation baseline for Table 2.

## Architecture preserved (unchanged)

- Backbone: **ResNet-101**, `output_stride=8`, with dilated convolutions in layer3/layer4
- Decoder: **DeepLabHeadV3Plus** -- ASPP (rates [12, 24, 36]) + low-level skip (layer1, 256 ch projected to 48 ch) + 304-ch classifier
- Pretrained backbone: ImageNet-1k (`pretrained_backbone=True`)
- Called via official `_load_model('deeplabv3plus', 'resnet101', num_classes=1, output_stride=8, pretrained_backbone=True)`

## Adaptations (task interface only)

| What | Official | Adapted |
|------|----------|---------|
| `num_classes` | 21 | **1** (binary rebar segmentation) |
| Output upsample | To input size (built into `_SimpleSegmentationModel.forward`) | Unchanged -- no extra resize added |
| Loss | Cross-entropy | `bce_dice_loss` via `train.py` |
| `pretrained_backbone` | True | True when `pretrained=True`, False when `pretrained=False` |

## Official repo files modified

**None.** `rebar_adapt/deeplabv3plus_rebar.py` inserts `repo/` into `sys.path` and
imports `network.modeling._load_model`. No file inside `repo/` is touched.

## Training command

```bash
python baselines_clean/train.py --mode deeplab \
    --data_root data --subsets site-1 site-2 \
    --img_h 480 --img_w 640 --epochs 50
```

## Inference command

```bash
python baselines_clean/predict.py --mode deeplab \
    --checkpoint checkpoints_clean/deeplab/best.pth \
    --data_root data --subsets site-1 site-2 --split val \
    --pred_dir predictions
```

## Output directory

```
predictions/deeplab/{subset}/seg/{id}.png
```

## Current status

- [x] MODIFICATIONS.md written
- [x] rebar_adapt/deeplabv3plus_rebar.py implemented (Stage D, revised Stage G)
- [ ] Smoke test passed
- [ ] Training run started
