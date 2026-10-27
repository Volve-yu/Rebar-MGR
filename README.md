# Rebar-MGR

## Public subset release

Only `public_data/` is distributed: 100 images from Site 1 and 400 from Site 2, with matching segmentation, edge, centerline and physical measurement records. Filenames use `工地一_0001` and `工地二_0001`. See `public_data/README.md` and `id_mapping.csv`.

Before using the experiment commands below, run `python scripts/prepare_public_data.py`. It creates ignored `data/` and `annotation_all_images/` directories in the legacy runner layout without changing image pixels or readings. The public splits contain 73/17 training/validation images for Site 1 and 317/79 for Site 2; 14 additional images retain unassigned status. These subsets cannot reproduce the full-data paper scores. Old full-data hashes, datasets and annotations have been moved outside this release. The layout and experiment instructions below describe the generated working directories, not additional publicly distributed data. `SOURCE_SNAPSHOT.json` is no longer included because it described the full private snapshot.

Rebar-MGR is a measurement-grounded rebar inspection and reporting pipeline. It combines:

1. edge-assisted and projection-guided multitask segmentation;
2. deterministic spacing extraction from a predicted mask and an independently recorded laser range;
3. design-rule screening and evidence-constrained report generation.

The released acquisition protocol uses an RGB camera in RGB mode and an independent laser ranging module. It does not require or use a depth map, RGB-D frame, stereo pair, or point cloud. The output is intended for preliminary human-supervised review and prioritization; it is not a surveying-grade measurement or a certified construction acceptance procedure.

## Repository layout

```text
src/                    baseline segmentation, measurement, and evaluation code
src_measure_guided/     EPM training, prediction, and measurement-guided losses
src_report/             structured evidence and report-generation utilities
measurement_song2026/   detector/Hough adaptation for 2D measurement comparison
scripts/                experiment runners and screening evaluation
baselines_clean/        unified external-baseline interface
data/                   two-site RGB images and annotations
annotation_all_images/  user-verified interval/design-rule table
tests/                  focused regression tests
SOURCE_SNAPSHOT.json    copied-file provenance (before packaging edits)
```

The folder does not include the manuscript, trained checkpoints, API keys, generated reports, experiment caches, or temporary archives. It includes locally used third-party sources and their available licenses. This packaging step does not grant or verify redistribution rights for those sources; see THIRD_PARTY.md. No new license has been assigned to the authors' code or data.

## Environment

Python 3.9 or newer is recommended. For the core pipeline:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch and torchvision build matching the CUDA version of the machine before or instead of the generic PyTorch requirement. Verify the installation with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

External legacy baselines use additional packages and, for Mobile-Seed, a separate environment. Use `requirements_external.txt` only after checking the official implementation and CUDA compatibility.

## Data format

The included data has two subsets, `site-1` and `site-2`:

```text
data/<site>/
  images/<id>.png
  segmentation/<id>.png
  edge/<id>.png
  rebar_labels/<id>.txt
  depth_labels.txt
  gt_sets/{train,val}.txt
```

`depth_labels.txt` contains the scalar camera-to-rebar-plane range recorded by the laser. It is a scalar calibration input, not a depth image. The interval table stores user-verified readings, design spacing, signed lower tolerance, and signed upper tolerance. Do not replace the design reference with the mean measured spacing.

## Train the proposed model

Run from this repository root. The following example trains the full EPM model on Site 1:

```bash
python src_measure_guided/train.py \
  --subsets site-1 --seed 42 --epochs 50 --batch_size 4 \
  --alpha_edge 0.10 --beta_proj 0.15 --lambda_c 0 \
  --max_edge_to_seg 0.04 --max_proj_to_seg 0.05 \
  --aux_schedule warmup_cosine_decay --aux_warmup_epochs 5 \
  --aux_min_scale 0.2 --seg_refine_epochs 5 \
  --save_dir runs/site1_full/checkpoints \
  --log_dir runs/site1_full/logs --device cuda
```

Predict and evaluate the validation split:

```bash
python src_measure_guided/predict.py \
  --checkpoint runs/site1_full/checkpoints/best.pth \
  --subsets site-1 --split val \
  --pred_dir runs/site1_full/predictions --device cuda

python src_measure_guided/eval.py \
  --checkpoint runs/site1_full/checkpoints/best.pth \
  --subsets site-1 --device cuda
```

For the complete within-site and bidirectional cross-site experiment plan, use a new output directory:

```bash
python scripts/run_revision_experiments.py \
  --run_root runs/revision_full_seed42 \
  --variants full --seeds 42 --execute
```

The runner refuses to overwrite an existing output directory. Omit `--execute` to generate `commands.json` and `run.sh` only, then execute `bash runs/revision_full_seed42/run.sh` on Linux. Do not invoke the plan generator again on the same output directory. Sites are trained separately; this is not joint training on both sites.

Run the loss ablation (two sites, three variants) with:

```bash
python scripts/run_experiment_three.py --plan-only
python scripts/run_experiment_three.py
```

For this ablation runner, use `--resume` after a failed stage. It reuses completed stages and resumes supported training checkpoints.

## Measurement and design-rule screening

The physical inspection runner calibrates peak parameters on training predictions and evaluates them on validation predictions. It records coverage and unavailable assessments rather than silently dropping unmatched intervals.

After the Full runner above finishes, the agreement experiment uses its frozen Full predictions and the user-verified interval table:

```bash
python scripts/run_screening_agreement.py
```

It writes a new timestamped directory under `results/` containing `summary.json`, `interval_details.csv`, `image_details.csv`, and `README.md`. The primary protocol uses the region-specific design rules. Uniform +/-10 mm and +/-20 mm rows are exploratory sensitivity analyses, not automatic engineering standards.

## Report generation

The deterministic report path can run without an API key. API-backed generation requires an environment variable and must not be committed:

```bash
# Linux/macOS
export DASHSCOPE_API_KEY='your_key_here'
# Windows PowerShell
$env:DASHSCOPE_API_KEY = 'your_key_here'
```

After the Full runner finishes, train QAEP on training evidence, apply it to validation evidence and evaluate deterministic reports without API calls:

```bash
python scripts/run_revision_followups.py 4 --source-root runs/revision_full_seed42 --run-root runs/reports --model qwen3.8-omni-flash
```

Then, after setting the environment variable, run all eight report settings (paid API calls):

```bash
python scripts/run_revision_followups.py 4 --source-root runs/revision_full_seed42 --run-root runs/reports --model qwen3.8-omni-flash --resume --run-api
```

The model alias is the one recorded in the revision experiment; availability depends on the service account. `python scripts/list_dashscope_models.py --help` describes model discovery. Do not silently substitute a different model for only some settings. Report scores measure preservation of supplied evidence, not independently verified physical accuracy.

For a short final inspection summary, add `--report-style concise` to a direct
`src_report/generate.py` command and use a new output file. This optional mode
aims for four sentences, rejects outputs exceeding six sentences or 100 words,
and retains rejected raw responses for inspection. The default `original` mode
preserves the prompts used for the reported experiments. Figure summaries do
not replace archived outputs or change their evaluation scores.

## Additional measurement and report comparisons

The added detector/Hough comparison is documented in `measurement_song2026/README.md`. `scripts/run_traditional_measurement.py` evaluates RGB-Canny-Hough, predicted-mask Hough, and the EPM projection routine under one physical-reference protocol. It requires Full-model train/validation predictions and the separately supplied physical-reference table; these full-data inputs are not included in this code update. The Hough settings are selected on the training split, not on validation data. For method comparisons on common recovered intervals, use `scripts/compare_traditional_common_intervals.py` with its `--help` options.

The report comparison utilities `scripts/compare_template_with_groundtruth.py` and `scripts/rescore_reports_with_groundtruth.py` accept explicit paths to generated reports and reviewed reference reports. `scripts/rebuild_reviewed_groundtruth.py` provides the separate draft, review-sheet and export workflow when physical-reference inputs are available; it does not automatically mark a draft as reviewed. Their outputs are written to caller-selected directories; neither source data nor generated reports are committed. `src_report/generate.py` already supports the measurement-only and measurement-plus-rule input settings. The complete paper scores require the private full validation set and its reviewed references; the public subset is for inspecting the implementation and running smaller independent tests.

## External baselines

The unified baseline runner supports the public-paper comparison interface:

```bash
python scripts/run_external_experiment_one.py \
  --methods deeplab gscnn sctransnet afenet \
  --sites site-1 site-2
```

These adapters standardize dataset input, binary output, prediction export, and downstream measurement. They are task adaptations and should not be described as exact reproductions of the cited papers. Official pretrained weights are intentionally excluded. Mobile-Seed requires its official ImageNet checkpoint and a separate compatible environment; consult the original repository and the accompanying license before downloading it.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover signed tolerance handling, interval matching, unavailable records, and report evidence safeguards. A full training run is not part of the smoke test.

## Reproducibility and data use

The included annotations are user-verified field records and have not been independently audited as a traceable metrology standard. Results should report the annotation provenance, matching coverage, interval-level errors, false-pass rate, and image-level class balance. Do not claim that a high image-level agreement alone establishes reliable PASS/FAIL discrimination when one class is rare.

The dataset may contain project-specific construction imagery. Confirm that the data owner permits public redistribution before pushing the `data/` directory to GitHub. If redistribution is not authorized, remove `data/` and publish the same directory layout, annotation schema, and a download or access procedure instead.

## Citation

The manuscript is titled "Rebar-MGR: An Edge-Segmentation-Measurement Coupled Multitask Framework with Evidence-Constrained Report Generation for Rebar Spacing Inspection". Publication DOI and finalized citation metadata are not included. This local folder has not been published to GitHub.
