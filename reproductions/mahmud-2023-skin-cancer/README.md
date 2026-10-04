# Mahmud et al. (ICCIT 2023): skin cancer categorization

Paper: **An Interpretable Deep Learning Approach for Skin Cancer Categorization**.
Authors: Faysal Mahmud, Md. Mahin Mahfiz, Md. Zobayer Ibna Kabir, and Yusha Abdullah.
Venue: 2023 26th International Conference on Computer and Information Technology (ICCIT), pages 1–6.
DOI: [10.1109/ICCIT60459.2023.10508527](https://doi.org/10.1109/ICCIT60459.2023.10508527).
Author repository: [Faysal-MD/An-Interpretable-Deep-Learning-Approach-for-Skin-Cancer-Categorization-IEEE2023](https://github.com/Faysal-MD/An-Interpretable-Deep-Learning-Approach-for-Skin-Cancer-Categorization-IEEE2023).
Bibliography and headline targets checked against the author repository on 2026-10-05.
Upstream commit and full paper methodology have not yet been pinned/verified.

## Imported notebooks

These are two independent experiments, not sequential pipeline stages. File names
indicate their roles; notebook contents, outputs, and execution counts are preserved.

| Notebook | Role | Existing saved outputs |
|---|---|---|
| [01_efficientnetv2s_reproduction_colab.ipynb](notebooks/01_efficientnetv2s_reproduction_colab.ipynb) | Targets the released EfficientNetV2S experiment: image-level stratified split, full-backbone training, 50 epochs, Faster Score-CAM | Test accuracy 89.5210%; Score-CAM cell contains `NameError: true_labels is not defined`; final evidence export has no execution recorded |
| [02_xception_lesion_safe_extension_colab.ipynb](notebooks/02_xception_lesion_safe_extension_colab.ipynb) | Modified Xception experiment: lesion-grouped split, streaming data, class weights, two-stage training, entropy/referral metrics, SmoothGrad | Test accuracy 66.8347%, macro F1 0.420228; evaluation and archive-generation outputs present |

These numbers are read from imported notebook outputs, not independently rerun or
validated. EfficientNetV2S has nonsequential execution counts and a saved error;
its outputs do not establish a clean top-to-bottom run. Xception uses a different
split and training protocol, so its accuracy is not a like-for-like paper replication.

The author repository reports 88.02% accuracy for EfficientNetV2S and 88.72% for
XceptionNet. The notebook's more precise EfficientNetV2S reference number has not
been independently verified in this organization pass.

## Dataset and scope

Both notebooks use HAM10000 from the Kaggle mirror
`kmader/skin-cancer-mnist-ham10000`; the authors link the Harvard Dataverse dataset.
Exact dataset version, checksums, access/license details, and upstream commit remain
to be recorded. Current scope covers EfficientNetV2S reproduction and a separate
Xception extension, not all four models in the author repository.

## Setup and execution

Open either notebook independently in Google Colab and follow its setup cells.
Both expect Colab paths under `/content/`, a GPU runtime, and Kaggle access.
Moving the files does not change these runtime paths. Training was not run locally.
Download the artifact bundles before the Colab runtime ends; keep large checkpoints
outside Git and copy small metrics/provenance files into `results/` under unique run names.
Record pinned dependencies and a reproducible environment before the next run.
Do not assume the original project's environment or checkpoints match the paper.

## Layout

- `configs/`: separate configurations for faithful reproduction and extensions.
- `src/`: implementation specific to this paper.
- `notebooks/`: exploration and result inspection.
- `results/`: small run records and comparison tables committed to Git.
- `deviations.md`: differences from the paper and their effect on comparability.
- `data/`, `checkpoints/`, `outputs/`: local artifacts ignored by Git.

## Experiment records

For each run, save a uniquely named JSON file in `results/` containing:
run ID, UTC timestamp, Git commit and dirty state, config path and hash, seed,
Python and dependency versions, hardware, dataset version/checksum, split manifest
path/checksum, preprocessing, training settings, checkpoint location/checksum,
metric definitions and measured metrics. Record failed runs and their failure reason.
Commit small split manifests when permitted; keep sensitive identifiers outside Git.
Use validation data for tuning and record when the test set is evaluated.

## Comparison

| Item | Paper | Reproduction | Comparable? / reason |
|---|---|---|---|
| Dataset and split | HAM10000; exact protocol still to verify | Image-level EfficientNetV2S; lesion-grouped Xception | Xception split differs |
| Preprocessing | Pending | Pending | Pending |
| Architecture and training | Pending | Pending | Pending |
| Metric definitions | Pending | Pending | Pending |
| Reported / measured results | EfficientNetV2S 88.02%; XceptionNet 88.72% (author README) | Saved outputs above | No fresh validation |

Keep extensions separate from the faithful reproduction and report both explicitly.
