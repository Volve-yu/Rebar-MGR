# Measurement-Guided Training Variant

This directory is separate from `src/` and does not modify old experiments.

It reuses:

- `src/model.py` (`RebarMTL`)
- `src/dataset.py`
- `src/eval.py`

It adds:

- `MeasurementGuidedLoss`
- differentiable column-projection geometry loss

## Rationale

Current method framing:

1. The input is an RGB image.
2. The network predicts both segmentation and boundary/edge maps.
3. The boundary branch is an auxiliary structural cue, not the main comparison
   target.
4. Segmentation is the primary task used for external baseline comparison.
5. Measurement is derived from predicted structural cues and is used here as a
   differentiable training cue to improve segmentation.

The downstream measurement module computes count/spacing from the horizontal
distribution of rebar masks. Direct count/spacing extraction is
non-differentiable, so this variant uses a differentiable surrogate:

```text
pred_col(x) = mean_y sigmoid(seg_logit(y, x))
gt_col(x)   = mean_y seg_gt(y, x)
```

If `rebar_labels/{id}.txt` is available, the existing dataset also provides a
soft center heatmap. Its column projection is used as an additional x-position
cue.

## Loss

The current implementation uses:

```text
L = L_seg + alpha_edge * L_edge + beta_proj * L_proj + lambda_c * L_cons
```

where:

- `L_seg`: binary segmentation BCE + Dice.
- `L_edge`: weighted edge BCE + Dice.
- `L_proj`: differentiable measurement-aware column projection loss.
- `L_cons`: optional edge/seg consistency loss.

Default weights:

```text
alpha_edge = 0.3
beta_proj  = 0.1
lambda_c   = 0.0
```

These defaults follow the current interpretation that edge supervision is
auxiliary. A lower `alpha_edge` reduces negative transfer from the edge branch
to the primary segmentation branch. `beta_proj` introduces measurement-aware
geometry without relying on non-differentiable post-processing during training.

## Train

```bash
python src_measure_guided/train.py \
  --epochs 50 \
  --device cuda \
  --alpha_edge 0.3 \
  --beta_proj 0.1 \
  --lambda_c 0.0
```

Outputs:

```text
checkpoints_measure_guided/best.pth
checkpoints_measure_guided/latest.pth
logs_measure_guided/run_*.json
```

## Suggested Ablation

Run only this new variant first:

```text
alpha_edge=0.3, beta_proj=0.1, lambda_c=0.0
```

Compare against the previous best non-measurement-guided variant. If it does not
improve validation segmentation or measurement stability, keep it as an
ablation instead of making it the final method.

Then optionally run:

```text
alpha_edge=0.15, beta_proj=0.1, lambda_c=0.0
alpha_edge=0.3,  beta_proj=0.05, lambda_c=0.0
alpha_edge=0.3,  beta_proj=0.1,  lambda_c=0.03
```

Do not claim direct optimisation of count/spacing MAE unless a differentiable
measurement formulation is implemented. The current claim should be
"measurement-guided geometric surrogate".
