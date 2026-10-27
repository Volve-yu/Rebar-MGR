# Two-dimensional measurement comparison

This directory contains the detector and Hough-transform adaptation used for the added measurement comparison. It trains YOLOv8n, YOLO11n, and a paper-guided Rebar-Light reconstruction separately for Site 1 and Site 2, then evaluates vertical-rebar center spacings using a common scalar laser range. The reconstruction is not an exact reproduction of the source paper's RGB-D system. No model weights, reviewed detection boxes, or full experimental dataset are distributed here.

Install PyTorch and torchvision for your platform, then install `measurement_song2026/requirements.txt`. Run from the repository root. The public subset can be staged with `python scripts/prepare_public_data.py`, but it does not reproduce the full-data results in the manuscript.

1. Generate detection-box drafts from the staged `data/` directory. Inspect and correct every box before marking its row as reviewed in `review.csv`.

   ```bash
   python measurement_song2026/prepare.py --data-root data --out song2026_detection_drafts
   python measurement_song2026/prepare.py --data-root data --out song2026_detection_drafts --confirm-reviewed
   ```

2. Train one detector with the reviewed boxes:

   ```bash
   python measurement_song2026/run.py train --data-root data --dataset song2026_detection_drafts \
     --site site-1 --model yolov8n --out runs/song2026_site1_yolov8n --device 0 --epochs 50 --batch 16
   ```

3. Evaluate using the separately supplied physical-reference annotation table and the verified camera horizontal field of view:

   ```bash
   python measurement_song2026/run.py evaluate --data-root data --site site-1 \
     --run runs/song2026_site1_yolov8n --out runs/song2026_site1_yolov8n/measurement \
     --annotations annotation_all_images/all_images_annotations.csv --hfov 87 --device 0
   ```

   The `87` degree value is the historical nominal setting, not a new camera calibration. Check the setting for your acquisition before evaluating. The same scale convention must be used for every method.

`run_all.sh` performs the six site/model runs after `DATA_ROOT`, `ANNOTATIONS`, and `HFOV` are set. Its outputs belong under `runs/`, not in Git. Evaluation reports matched-spacing error together with interval coverage and failed detection cases; low error alone is not sufficient when intervals are missing.
