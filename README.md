# ASU Image Processing Warmups

> **TL;DR:** Commented Python demos for CSE 507 image-processing warmups 1–9. **Inputs:** the educational dataset ZIPs listed below. **Outputs:** image conversions, training/evaluation figures, logs, metrics, and annotations.

This collection progresses from medical-image I/O through supervised prediction, segmentation, localization, and unsupervised learning.

[Read the anonymized chest X-ray report (PDF)](reports/Chest_Xray_Warmup_Report_Anonymized.pdf).

## Demo guide

All measurements below are historical experiment results, not outcomes of a fresh run of these Python conversions.

| Warmup | What the demo does | Required inputs | Expected outputs / reported result |
| --- | --- | --- | --- |
| 1 | Read, display, and write PNG/JPEG/DICOM chest X-rays | `Practice_PNGandJPG.zip`, `Practice_DICOM.zip` | Format-preserving image copies, DICOM copy and 8-bit preview, comparison figures; source images are 2048×2048 |
| 2 | Classify up/right/down/left with pretrained ResNet-18 and a trained replacement head | `Directions01.zip`; 948 train / 40 test images | Metrics CSV, accuracy plot, final weights; reported 100% test accuracy (40/40) |
| 3 | Fine-tune ResNet-18 for the supplied binary gender labels | `Gender01.zip`; 154 train / 93 test | Accuracy curve and log; reported 87.1% (81/93) |
| 4 | Estimate age with a single-output ResNet-18 regression head | `XPAge01_RGB.zip`; 80 train / 165 test; CSV columns `filenames`, `age` | MAE curve/log in years; best observed 8.52 at epoch 20, final 9.36 at epoch 60 |
| 5 | Train a binary U-Net for lung semantic segmentation | `Segmentation01.zip`; 50 train / 10 test; lung value 255 | Dice curve and qualitative prediction; reported best source Dice 0.872, final 0.837 |
| 6 | Train a multiclass U-Net for anatomical regions | `Segmentation02.zip`; 199 train / 48 test | Foreground Dice curve and prediction figure; reported 0.882 at epoch 25 |
| 7 | Convert supplied masks into bounding boxes with connected components | The 48 Segmentation02 test masks | Bounding-box CSV and visualization; reported 193 boxes: heart 48, chest/body 49, lungs 96 |
| 8 | Detect novelty through autoencoder reconstruction error | `autoencoder_img.zip`; 200 normal training, 30 normal and 17 flipped evaluation | Reconstruction/score figures and log; ROC-AUC 0.551, threshold 0.011719 |
| 9 | Cluster 64-dimensional autoencoder embeddings with K-means | `Directions01.zip` and `autoencoder_img.zip` | Cluster CSVs, PCA plots, mosaic; orientation ARI/NMI 1.000/1.000; normal/flip −0.010/0.015 |

**Scope:** Warmups 1–9. The additional MNIST ResNet-18 notebook is included as a separate **prewarmup**, consistent with its local course folder.

## Python demos

Each Python file opens with a TL;DR, inputs, and expected outputs, followed by comments explaining preprocessing, training, evaluation, and persistence. Related exercises remain together because the original notebooks share utilities and experiment state.

| Python file | Coverage |
| --- | --- |
| [warmup_01_image_io.py](warmups/warmup_01_image_io.py) | Exercise 1 |
| [warmup_02_orientation.py](warmups/warmup_02_orientation.py) | Exercise 2 |
| [warmups_03_04_classification_regression.py](warmups/warmups_03_04_classification_regression.py) | Exercises 3–4, in order |
| [warmups_05_07_segmentation_localization.py](warmups/warmups_05_07_segmentation_localization.py) | Exercises 5–7, in order |
| [warmups_08_09_autoencoder_clustering.py](warmups/warmups_08_09_autoencoder_clustering.py) | Exercises 8–9, in order; 9 reuses the normal-image autoencoder embeddings |
| [prewarmup_resnet18_mnist.py](warmups/prewarmup_resnet18_mnist.py) | Additional MNIST experiment; downloads MNIST and prints losses/accuracy |

## Run locally

Use Python 3.10 or newer. Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the required archives inside `data/`. Dataset archives, extracted medical images, and model checkpoints are not distributed here. Obtain the course datasets separately. The scripts retain the supplied train/test folder structure; do not flatten the archives.

```bash
python warmups/warmup_01_image_io.py --data-dir ./data --output-dir ./outputs
python warmups/warmup_02_orientation.py --data-dir ./data --output-dir ./outputs
python warmups/warmups_03_04_classification_regression.py --data-dir ./data --output-dir ./outputs
python warmups/warmups_05_07_segmentation_localization.py --data-dir ./data --output-dir ./outputs
python warmups/warmups_08_09_autoencoder_clustering.py --data-dir ./data --output-dir ./outputs
python warmups/prewarmup_resnet18_mnist.py --output-dir ./outputs
```

Every script supports `--help` without requiring the ML dependencies to be installed. `--work-dir` optionally selects a dedicated extraction/download directory; otherwise scratch data is under `outputs/work/<script-name>`. Repeated runs overwrite matching result files. Exercise 1 replaces its own extraction subfolders, and Exercise 2 replaces its dedicated scratch folder. Choose scratch paths used only for this demo. Other scripts reuse existing extracted folders; use a fresh scratch directory when changing archives.

A CUDA GPU is recommended; the scripts fall back to CPU. Pretrained ResNet-18 weights and MNIST require network access on their first download. Figures use a noninteractive backend and are saved to disk. The original epoch counts and learning settings remain in the scripts. Dataset workers use a single process for compatibility across macOS, Windows, and Linux. Dependencies list direct requirements, not a historical lockfile; platform-specific PyTorch installation may be needed.

## Data expectations

- **Orientation:** `Directions01/train/<class>` and `Directions01/test/<class>`.
- **Gender labels:** `train/male`, `train/female`, and matching `test` folders.
- **Age:** `trainingdata.csv` and `testdata.csv`, with image filenames matching the `filenames` column and numeric ages in `age`.
- **Lung masks:** `train/org`, `train/label`, `test/org`, `test/label`.
- **Organ masks:** `org_train`, `label_train`, `org_test`, `label_test`. The `_label` suffix is removed when pairing stems across image formats. Raw mask values are 0 background, 85 heart, 170 chest/body, 255 lung.
- **Autoencoder:** `autoencoder_img/normal200`, `autoencoder_img/normal`, and `autoencoder_img/flip`.

## Interpretation and limitations

Exercise 6 implements multiclass **semantic** segmentation, despite the original assignment's “instance segmentation” wording. Exercise 7 derives boxes from ground-truth masks, so the box count is not an object detector accuracy measurement. Exercise 5's source Dice function averages present background and foreground classes; Exercise 6 averages present non-background classes. Preserve that distinction when comparing scores.

The age result of 8.52 years was selected from repeated test evaluations and is optimistic as a final test estimate. The original run did not preserve an epoch-20 checkpoint. Most demos save figures/logs rather than model weights; only the orientation script saves a state dictionary. These small instructional experiments are educational baselines, with no clinical performance claim. High orientation accuracy and clustering agreement do not overcome the weak normal/flip novelty result.

## Provenance and verification

`notebooks/` contains the selected complete Colab notebook code, with a new input/output summary. Stored outputs and runtime metadata were cleared for publication; the original downloaded files were left untouched. The incomplete exercises 5–7 duplicate was excluded. [Source checksums](docs/source-manifest.json) identify the original notebook inputs.

The Python conversions remove Colab mounting and notebook package-install commands, expose directory arguments, replace notebook-only table display, support CPU execution, format the code, and guard constant-image DICOM preview scaling. An anonymized research report is included.

Verification: Python syntax, undefined-name checks, import safety, and each command's `--help` passed. Exercise 1 also passed a synthetic end-to-end smoke run: all six expected files were produced, PNG and DICOM pixels were preserved, and a constant-intensity DICOM produced a valid preview. Training and full dataset-dependent execution have **not** been rerun. Historical notebook outputs and metrics are not substitutes for fresh reproduction.

## References

Methods follow He et al. (ResNet, CVPR 2016), Ronneberger et al. (U-Net, MICCAI 2015), Hubert and Arabie (adjusted Rand index, 1985), and Strehl and Ghosh (NMI/cluster ensembles, JMLR 2002). Educational chest radiographs derive from JSRT miniJSRT.
