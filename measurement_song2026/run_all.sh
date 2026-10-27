#!/usr/bin/env bash
set -euo pipefail
# Run from the code/project root containing measurement_song2026/ and src/.
: "${DATA_ROOT:?Set DATA_ROOT to the existing data directory}"
: "${ANNOTATIONS:?Set ANNOTATIONS to all_images_annotations.csv}"
: "${HFOV:?Set HFOV to the camera horizontal field of view in degrees}"
BOX_ROOT="${BOX_ROOT:-song2026_detection_drafts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/song2026_measurement}"
for site in site-1 site-2; do
  for model in yolov8n yolo11n rebar_light; do
    run="$OUTPUT_ROOT/${site}_${model}_seed42"
    if [ ! -f "$run/experiment.json" ]; then
      python -u measurement_song2026/run.py train \
        --data-root "$DATA_ROOT" --dataset "$BOX_ROOT" --site "$site" \
        --model "$model" --out "$run" --device 0 --epochs 50 --batch 16
    fi
    if [ ! -f "$run/measurement/protocol.json" ]; then
      python -u measurement_song2026/run.py evaluate \
        --data-root "$DATA_ROOT" --annotations "$ANNOTATIONS" --site "$site" \
        --run "$run" --out "$run/measurement" --device 0 --hfov "$HFOV"
    fi
  done
done
