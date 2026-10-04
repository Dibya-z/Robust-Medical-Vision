# Robust Medical Vision

This repository contains the original HAM10000 uncertainty-aware classification
project (DermaSense AI) and a separate workspace for reproducing a research paper.

| Work | Location | Status |
|---|---|---|
| Original project | `ML/`, `DL/`, `Final/`, `web/` | Existing implementation and previously reported results |
| Paper reproduction | [Reproduction workspace](reproductions/mahmud-2023-skin-cancer/README.md) | Imported Colab notebooks; reproduction and extension documented |

## Original project

See [original project documentation](docs/original-project.md) for the existing
methodology, reported metrics, and application setup. Those metrics belong to the
original project and have not been revalidated as part of the reproduction.

- `ML/`: classical machine learning baseline.
- `DL/`: earlier deep learning workflow; its README describes EfficientNet-B1.
- `Final/`: integrated training, calibration, OOD, and conformal experiments;
  `models/architecture.py` uses B1 and `models/architecture_v2.py` uses B3.
  Check the run configuration
  and checkpoint for the actual architecture used in any result.
- `web/`: existing FastAPI and React demo.
- `PROJECT_ACHIEVEMENTS.md`: existing portfolio notes, separate from reproduction evidence.

The original project is an experimental system; its documentation does not establish
clinical validation or suitability for medical decisions.

## Repository maintenance

Keep the original code paths stable. Add paper-specific implementation under
`reproductions/<paper-short-name>/`; the current paper lives in `reproductions/mahmud-2023-skin-cancer/`.
Develop on a branch such as `reproduce/<paper-short-name>` and merge reviewable work
into the main branch. Keep both efforts accessible in the repository after merging.

Before changing the original implementation, commit the original files intended
for preservation and tag that commit `original-project-v1`. A tag preserves committed
files only; separately preserve ignored datasets, environments, and checkpoints.
The pre-scaffold committed baseline is `edce3ae`; the previously untracked
`PROJECT_ACHIEVEMENTS.md` is not part of that commit.

For each reproduction, document the paper version, environment, data/splits,
commands, run records, deviations, and paper-versus-reproduced results. Keep faithful
reproduction and extensions in separate configurations. Never overwrite old runs.
Commit small metrics and provenance files; keep large data and checkpoints out of Git.
