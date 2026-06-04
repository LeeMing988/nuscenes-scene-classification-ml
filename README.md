# nuscenes-scene-classification-ml

![Python](https://img.shields.io/badge/python-3.12.10-blue)
![Methodology](https://img.shields.io/badge/methodology-5--fold%20scene--aware%20CV-green)
![Stages](https://img.shields.io/badge/study-two--stage%20(mini%20%E2%86%92%20150--scene)-blueviolet)
![License](https://img.shields.io/badge/license-Academic-orange)
![Status](https://img.shields.io/badge/status-Complete-success)

A comparative study of classical machine-learning models for **multi-attribute scene classification** of front-camera driving images on the [nuScenes](https://www.nuscenes.org/) autonomous-driving dataset.

| Field | Details |
|---|---|
| **Author** | YEOH LEE MING |
| **Student ID** | TP118013 |
| **Course** | Applied Machine Learning (Master in AI) |
| **Institution** | Asia Pacific University of Technology & Innovation (APU) |
| **Date** | June 2026 |

---

## Project overview

Autonomous-driving systems must continuously interpret their operating context — time of day, weather, surrounding traffic density, and the presence of vulnerable road users (VRUs). While the nuScenes literature focuses overwhelmingly on deep 3D object detection and tracking, **holistic scene-level attribute classification** using classical machine learning is comparatively unexplored.

This project asks whether classical ML on hand-crafted visual features can reliably classify four operational scene attributes from a single front-camera image, **and how dataset scale affects both performance and evaluation validity**. Five classical model families are compared across four independent classification tasks, evaluated with 5-fold scene-aware cross-validation to prevent scene leakage.

### Two-stage design

The study is deliberately structured as **prototype → primary experiment**:

| Stage | Dataset | Role |
|---|---|---|
| **Stage 1** | `v1.0-mini` (10 scenes, 404 keyframes) | **Pilot / methodology development** — EDA, label design, feature engineering, dimensionality & preprocessing ablations, feature ablation. Establishes and justifies the pipeline. Found to be **too small for reliable evaluation** (see Findings). |
| **Stage 2** | `v1.0-trainval` 150-scene subset (6,021 keyframes) | **Primary experiment** — the validated pipeline applied at scale. **Conclusions are drawn here.** |

Stage 1 surfaced that 10 scenes are insufficient (degenerate weather, extreme cross-fold variance), which motivated extracting a 150-scene subset from the full `v1.0-trainval` release for a trustworthy evaluation.

### The four scene attributes

| # | Attribute | Classes | Source |
|---|---|---|---|
| 1 | `time_of_day` | day / night | Scene description text |
| 2 | `weather` | clear / rain | Scene description text |
| 3 | `vehicle_density` | low / medium / high | Tertile bins on forward-cone vehicle count |
| 4 | `vru_present` | absent / present | Forward-cone VRU annotation count ≥ 1 |

> Four **independent** classification tasks on shared images ("multi-attribute"), each solved separately with per-attribute model selection — not a single multi-output model.

### The five classical model families

| # | Model | Family | Decision boundary |
|---|---|---|---|
| 1 | Logistic Regression | Linear / probabilistic | Hyperplane |
| 2 | SVM (RBF kernel) | Kernel-based | Curved, margin-maximising |
| 3 | Random Forest | Tree ensemble (bagging) | Axis-aligned splits, averaged |
| 4 | XGBoost | Tree ensemble (boosting) | Sequentially corrected splits |
| 5 | MLP (1 hidden layer) | Shallow neural network | Non-linear, fully-connected |

### Research questions

- **RQ1.** Can hand-crafted visual features support multi-attribute scene classification on nuScenes front-camera images?
- **RQ2.** Which scene attributes are most and least learnable from a single front-camera frame?
- **RQ3.** Among classical model families, which performs most consistently across attributes?
- **RQ4.** Which feature types (HOG, colour histograms, LBP, photometric statistics) are most informative per attribute?
- **RQ5.** How much does hyperparameter tuning (and class weighting, where supported) improve over base classifiers?

---

## Key findings

> All headline numbers below are from the **150-scene subset (Stage 2)** — the trustworthy primary experiment. Macro-F1, best tuned model per attribute, mean across 5 folds × 3 seeds.

| Attribute | Best model macro-F1 | Random baseline | Majority baseline | Verdict |
|---|---|---|---|---|
| `time_of_day` | **~0.99** | 0.45 | 0.44 | Excellent — photometric signal |
| `weather` | **~0.90** | 0.45 | 0.44 | Strong — first valid weather result |
| `vehicle_density` | **~0.51** | 0.33 | 0.17 | Above baseline; structural ceiling |
| `vru_present` | **~0.63** | 0.50 | 0.35 | Above baseline; structural ceiling |

**All four attributes beat both random and majority baselines** — genuine learning throughout. The pattern is the core finding:

1. **Photometric attributes** (time-of-day, weather) — global brightness/colour/haze cues are captured well by classical holistic features; performance is strong and stable.
2. **Structural attributes** (vehicle density, VRU presence) — reach a genuine ceiling (~0.5–0.63) because holistic features cannot count objects or localise small/distant VRUs. This ceiling holds across **all five models**, indicating it is a property of the **features**, not the classifier.
3. **Dataset scale governs evaluation validity.** Stage 1 (10 scenes) produced unreliable estimates — 40–66% of pilot fits scored a perfect macro-F1 of 1.000 (trivially separable tiny folds), and weather was degenerate (rain confined to a single scene). The 150-scene subset enables trustworthy, tight-variance measurement. **Scaling did not raise performance; it revealed the truth.**

Per-attribute model-family differences are confirmed statistically (Friedman test, all four attributes significant; Nemenyi post-hoc with critical-difference diagrams) — supporting **attribute-specific model selection** rather than one universally best classifier.

---

## Repository structure

```
nuscenes-scene-classification-ml/
├── data/
│   ├── nuscenes/                              # raw dataset (NOT in repo — see Setup)
│   │   ├── v1.0-mini/                         # Stage 1 (maps, samples, sweeps, metadata)
│   │   └── v1.0-trainval/                     # Stage 2 (CAM_FRONT subset + metadata)
│   └── processed/
│       ├── v1.0-mini/{metadata,labels,features,splits}/
│       └── v1.0-trainval/{metadata,labels,features,splits}/
│
├── notebooks/
│   ├── v1.0-mini/                             # Stage 1 — pilot & methodology development
│   │   ├── 01_eda.ipynb
│   │   ├── 02_attribute_labels.ipynb
│   │   ├── 03_image_features.ipynb
│   │   ├── 04_splits.ipynb
│   │   ├── 05a_dim_reduction.ipynb            # PCA / LDA analysis
│   │   ├── 05b_preprocessing_ablation.ipynb   # contrast-enhancement ablation
│   │   ├── 06_classical_models.ipynb          # 5 models × base+tuned × 4 attrs × 3 seeds × 5 folds
│   │   ├── 07_feature_ablation.ipynb          # feature-group ablation
│   │   ├── 08a_headline_results.ipynb         # pilot headline tables
│   │   ├── 08b_visual_analysis_and_stats.ipynb
│   │   └── 08c_deep_analysis.ipynb
│   │
│   └── v1.0-trainval/                         # Stage 2 — primary experiment & conclusions
│       ├── 00a_metadata_eda.ipynb
│       ├── 00b_scene_selection.ipynb          # select 150 scenes
│       ├── 00c_extract_images.ipynb           # extract CAM_FRONT subset
│       ├── 01_attribute_labels.ipynb
│       ├── 02_image_features.ipynb
│       ├── 03_splits.ipynb                    # 5-fold scene-aware CV
│       ├── 04_classical_models.ipynb          # 600 fits (5×4×2×3×5)
│       ├── 05_compare_mini_vs_subset.ipynb    # cross-stage comparison
│       ├── 06a_headline_results.ipynb         # primary headline tables
│       ├── 06b_visual_analysis_and_stats.ipynb # confusion matrices, ROC, stats
│       ├── 06c_deep_analysis.ipynb            # feature importance, error analysis, conclusions
│       └── 06d_nemenyi_posthoc.ipynb          # Friedman verification, Nemenyi post-hoc, CD diagrams
│
├── models/
│   ├── v1.0-mini/fold_<k>/<attr>/             # pilot model checkpoints
│   └── v1.0-trainval/<attr>/                  # 600 model checkpoints (NOT in repo — see Setup)
│
├── results/
│   ├── v1.0-mini/{metrics,predictions,figures,final}/
│   ├── v1.0-trainval/{metrics,predictions,figures,final}/
│   └── comparison/{metrics,figures,final}/    # cross-stage comparison outputs
│
├── .gitignore
├── requirements.txt
├── folder_structure.txt
├── LICENSE
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.12.10 (tested)
- CPU only — no GPU required for classical ML
- Multi-core CPU recommended (grid search benefits from parallelism)
- Disk: ~4 GB for Stage 1 (mini); ~2 GB for Stage 2 reproduction (metadata + extracted subset)

### Installation
```bash
git clone https://github.com/LeeMing988/nuscenes-scene-classification-ml.git
cd nuscenes-scene-classification-ml

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

### Dataset

The **full** nuScenes dataset is **not redistributed** in this repository (licensing). Obtain it from the official source. A small, pre-extracted CAM_FRONT subset is shared via the link below **for assessment and reproducibility only**, consistent with the nuScenes non-commercial terms of use (see License).

**Stage 1 — v1.0-mini (full, ~4 GB):**
1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download), accept the terms.
2. Download **`v1.0-mini`** and extract to `data/nuscenes/v1.0-mini/` (contains `maps/`, `samples/`, `sweeps/`, metadata).

**Stage 2 — 150-scene subset (no 380 GB download needed):**
The full `v1.0-trainval` blobs are ~380 GB. To reproduce Stage 2 **you do not need them** — only the metadata and the extracted front-camera subset:
1. From official nuScenes, download **`v1.0-trainval` metadata only** (the small `_meta` archive, a few hundred MB) → `data/nuscenes/v1.0-trainval/v1.0-trainval_meta/`.
2. Download the **pre-extracted CAM_FRONT subset** (~1 GB, 150 scenes) → [Google Drive](https://drive.google.com/file/d/1YhC5dVxgEHdFl0MSIM2ucA0pem2PYehK/view?usp=sharing) → place at `data/nuscenes/v1.0-trainval/samples/CAM_FRONT/`.

> To regenerate the subset from scratch instead, download the full `v1.0-trainval` blobs and run `notebooks/v1.0-trainval/00a → 00b → 00c`.

---

## How to run

> **You do not need to retrain to reproduce the results.** All metrics, predictions, and figures are committed under `results/`. The analysis notebooks (`05`, `06a–d`, `08a–c`) read these saved outputs and regenerate every table and figure in minutes. Full model training (~38 h for Stage 2) is only needed to rebuild the models themselves.

### Stage 1 — v1.0-mini (pilot)
Run in order: `01 → 02 → 03 → 04 → 05a → 05b → 06 → 07 → 08a → 08b → 08c`.

### Stage 2 — v1.0-trainval (primary)
Run in order: `00a → 00b → 00c → 01 → 02 → 03 → 04 → 05 → 06a → 06b → 06c → 06d`.

| Notebook | Approx. runtime | Note |
|---|---|---|
| feature extraction (`03`/`02`) | minutes | HOG+colour+LBP+photometric → 6,216-dim vectors |
| `04_classical_models` (Stage 2) | **~38 h** | 600 fits; SVM-RBF tuned is the bottleneck. **Skip if using committed results.** |
| `05`, `06a–d` | minutes | read saved CSVs/models; regenerate all tables & figures |

> **Seeds for reproducibility:** `SEED_LIST = [42, 7, 123]`, `SPLIT_SEED = 42`.

---

## Methodology highlights

### 5-fold scene-aware cross-validation
Adjacent keyframes within a scene are near-duplicates; splitting them across train/test would leak information. Whole **scenes** are assigned to folds (stratified on `time_of_day`), so no scene spans the train/test boundary. Every keyframe is evaluated exactly once across the 5 folds. Single-class (fold, attribute) combinations are skipped and documented in `skipped_combos.json`.

### Other choices
- **Forward-cone filter:** vehicle/VRU counts use a ±30°, 50 m cone matching an ADAS front-camera field of view.
- **Train-only scaling:** `StandardScaler` fitted on the training fold only — no test leakage.
- **Class imbalance:** handled at the **algorithm level** via `class_weight='balanced'` for Logistic Regression, SVM-RBF, and Random Forest, where it is directly supported; XGBoost and MLP were guided by macro-F1 and baseline comparison instead. Algorithm-level weighting (where supported) was chosen over data-level resampling (SMOTE) — resampling cannot manufacture minority-class diversity from a single source scene.
- **Trivial baselines:** random and majority-class baselines anchor every comparison.
- **Base vs tuned:** all five models run with defaults (base) and grid-searched (tuned) via inner 3-fold CV; class weighting applies to Logistic Regression, SVM-RBF, and Random Forest only.
- **Significance testing:** base-vs-tuned compared with paired Wilcoxon signed-rank tests and Cohen's *d*; model-family differences per attribute tested with the **Friedman test** and **Nemenyi post-hoc** analysis (critical-difference diagrams generated in `06d`). Cross-stage comparison is reported descriptively (different datasets → unpaired).

---

## Experimental scale (verified)

| Component | Stage 1 (mini) | Stage 2 (subset) |
|---|---|---|
| Scenes / keyframes | 10 / 404 | 150 / 6,021 |
| Model fits (`all_metrics.csv` rows) | 1,140 | 600 |
| Skipped (fold, attribute) combos | 1 (weather, fold 0) | 0 |
| Test predictions saved | 46,050 | 722,520 |
| Scene leakage detected | 0 | 0 |

---

## Reproduction tiers

| Goal | What you need | Time |
|---|---|---|
| Regenerate all tables & figures | committed `results/` + run `05`/`06a–d` | minutes |
| Inspect / re-predict with trained models | full trained-models tree, both stages ([Google Drive](https://drive.google.com/file/d/1bRWjwt0jVbAKTIKPWdCyhZPava8o_Rrt/view?usp=sharing)) | minutes |
| Rebuild models from scratch | metadata + image subset + run `04` | ~38 h |

> **Where the models go:** the archive is the full `models/` folder (both stages). Unzip so you end up with `models/v1.0-mini/…` and `models/v1.0-trainval/<attr>/…`, matching the *Repository structure* above — **not** under `data/nuscenes/`, which holds only images and metadata. If the archive expands to a top-level `models/` folder, unzip at the repo root so it merges into the existing `models/`; if it expands directly to `v1.0-mini/` and `v1.0-trainval/`, unzip into `models/`. Either way, make sure you don't end up with `models/models/…`.

---

## Key limitations

- **Stage 1 scale:** 10 scenes are insufficient for reliable evaluation — pilot scores are inflated/unstable (single-scene rain makes weather degenerate; structural attributes show extreme cross-fold variance). This motivated Stage 2 and is itself a reported finding.
- **Structural-attribute ceiling:** HOG/LBP/colour features cannot count objects or localise small VRUs, capping vehicle-density and VRU-presence performance regardless of model or scale.
- **Single-frame restriction:** no temporal context across frames.
- **Compute:** SVM-RBF with probability calibration is the training bottleneck (~minutes per tuned fit at scale).

---

## Reproducibility

| Item | Where |
|---|---|
| Random seeds | `SEED_LIST = [42, 7, 123]`, `SPLIT_SEED = 42` (each notebook) |
| Label thresholds | `data/processed/<version>/labels/attribute_thresholds.json` |
| Feature spec | `data/processed/<version>/features/feature_metadata.json` |
| Fold assignments | `data/processed/<version>/splits/kfold_metadata.json` |
| Saved metrics & predictions | `results/<version>/metrics/`, `results/<version>/predictions/` |
| Library versions | `requirements.txt` |

---

## License
Academic project, provided as-is for educational purposes. The nuScenes dataset is licensed separately — see [nuscenes.org/terms-of-use](https://www.nuscenes.org/terms-of-use).

---

## Citation
```bibtex
@misc{yeoh2026nuscenes,
  author = {Yeoh, Lee Ming},
  title  = {Multi-Attribute Scene Classification of Front-Camera Driving Images:
            A Comparative Classical Machine Learning Study on the nuScenes Dataset},
  year   = {2026},
  institution = {Asia Pacific University of Technology and Innovation},
  url    = {https://github.com/LeeMing988/nuscenes-scene-classification-ml}
}
```

---

## Acknowledgements
- nuScenes dataset by Motional (formerly nuTonomy).
- Course staff at APU for guidance and feedback.
