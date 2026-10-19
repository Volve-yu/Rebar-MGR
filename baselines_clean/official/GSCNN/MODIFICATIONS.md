# MODIFICATIONS.md -- GSCNN (gscnn)

## Source

| Field | Value |
|-------|-------|
| Type | External baseline |
| Paper | "Gated-SCNN: Gated Shape CNNs for Semantic Segmentation" (Takikawa et al., ICCV 2019) |
| Official GitHub URL | https://github.com/nv-tlabs/GSCNN |
| Commit | edd16c5d |
| Clone date | 2026-05-02 |
| Dirty (uncommitted changes in repo/) | 0 |
| License | CC BY-NC-SA 4.0 |

## Table name

GSCNN (WideResNet-38)

## Purpose

Representative dual-task (seg + edge) baseline for Table 2.

## Architecture preserved (unchanged)

- Backbone: **wider_resnet38_a2** (`classes=1000, dilation=True`) -- random init
  (the official repo does not bundle pretrained backbone weights; random init is correct)
- Shape stream: `res1/res2/res3` BasicBlocks + `GatedSpatialConv2d` gates (`gate1/gate2/gate3`)
- Deep supervision: `dsn3`, `dsn4`, `dsn7` 1x1 convolutions
- Canny branch: `cv2.Canny(img, 10, 100)` CPU numpy round-trip; non-differentiable; preserved as-is
- ASPP: `_AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)` with edge input
- Decoder: `bot_fine` (m2 skip, 128->48) + `bot_aspp` (1280+256->256) + `final_seg` (256+48->num_classes)
- `num_classes=1` passed to constructor (original: 19 for Cityscapes)

## Adaptations (interface only)

| What | Official | Adapted |
|------|----------|---------|
| `num_classes` | 19 | **1** (binary rebar segmentation) |
| `criterion` | passed in, called in training branch | **None** -- never instantiated |
| `forward()` output | training: `criterion(...)` result; eval: `(seg_out, edge_out)` | **always `(edge_logit, seg_out)`** |
| `.cuda()` in Canny block | `torch.from_numpy(canny).cuda()` | **`.to(x.device)`** -- device-compatibility adaptation |
| edge output convention | `edge_out` is post-sigmoid `[0,1]` | inverse-sigmoid applied to produce raw logit for `edge_loss` |

## Device-compatibility adaptation (DCA) detail

`repo/network/gscnn.py` line 287 hardcodes `.cuda()` on the Canny tensor.
The adapter's `forward()` replicates the full official forward body verbatim,
replacing only that single line with `.to(x.device)`. All feature-computing
statements are otherwise identical and delegate to `self._gscnn.<layer>`.
This is a device-compatibility adaptation, not an architecture change.

## Official repo files modified

**None.**

## Training command

```bash
python baselines_clean/train.py --mode gscnn \
    --data_root data --subsets site-1 site-2 \
    --img_h 480 --img_w 640 --epochs 50
```

## Inference command

```bash
python baselines_clean/predict.py --mode gscnn \
    --checkpoint checkpoints_clean/gscnn/best.pth \
    --data_root data --subsets site-1 site-2 --split val \
    --pred_dir predictions
```

## Output directory

```
predictions/gscnn/{subset}/seg/{id}.png
predictions/gscnn/{subset}/edge/{id}.png
```

## Current status

- [x] MODIFICATIONS.md written
- [x] rebar_adapt/gscnn_rebar.py implemented (Stage H1)
- [ ] Smoke test passed
- [ ] Training run started
