# Local packaging checks

## Public subset update

The checks below originally described the full local package. That dataset and its full annotation table have now been moved outside the release to `private_release_backup_20261011`. The current `public_data` contains only 100 Site 1 and 400 Site 2 images and 4173 spacing rows. RGB pixels were compared exactly against the annotation source images before pairing. See `public_data/validation.json` for current counts. Generated `data/` copies use only the selected public images and are ignored by Git. The previous full-data SOURCE_SNAPSHOT.json is outside the release as well.

- Data validation passed: Site 1 train 302 / val 69; Site 2 train 519 / val 130.
- All active split RGB, segmentation and edge PNG files decoded successfully.
- No train/validation ID overlap within a site; scalar distance and interval references are present.
- 10 focused unit tests passed in this release directory.
- Ablation and external baseline plan-only commands passed without training.
- Source credential-pattern scan found no matching API token strings; token values were not printed. This is a limited pattern scan, not a security certification.
- Screening runner paths were adapted from the revision workspace layout to the release root. Other core algorithms and dataset bytes were retained.
- `SOURCE_SNAPSHOT.json` records the copied source hashes before documented packaging edits; it is not a final-file checksum manifest.
- Full GPU retraining, legacy Mobile-Seed installation, and paid report API requests were not repeated during packaging.
- No manuscript or figures are included. No remote GitHub repository was created, and no files were pushed.
- Original pretrained weights are excluded and may need downloading before training. See the official baseline READMEs and Chinese usage instructions.

The sibling directory `rebar-mgr-github-release` was an intermediate copy, not this deliverable. Its removal was blocked by automatic approval review. Use only `Rebar-MGR-GitHub` for the prepared repository.
