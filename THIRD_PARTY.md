# Third-party sources

The baseline directories are copies of the sources and local dataset adapters used in the revision workspace. Existing LICENSE files are preserved; this package does not relicense upstream work or establish permission where no license was present.

- DeepLabV3Plus: inspect `baselines_clean/official/DeepLabV3Plus/repo/README.md` and LICENSE.
- GSCNN: https://github.com/nv-tlabs/GSCNN
- SCTransNet: https://github.com/xdFai/SCTransNet
- AFENet: inspect its included README and source notices.
- Mobile-Seed: https://github.com/WHU-USI3DV/Mobile-Seed
- Report task adapters: `supplement_adapters/`, derived from the local adaptations associated with https://github.com/Ed1sonChen/Clip2Safety and https://github.com/LouisChen15/ConstructionSite-10k-Implementation . These are adapted evidence-access protocols, not full native reproductions.

Model weights, original repository Git history, build binaries and caches are not included. Historical baseline modes exposed by shared scripts may reference implementations not bundled here; use the five documented external baselines and `src_measure_guided` for the current Full model.
