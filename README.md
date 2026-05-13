# Multi-Attribute Scene Classification on nuScenes

A comparative study of classical machine-learning models for multi-attribute scene classification of front-camera driving images on the [nuScenes](https://www.nuscenes.org/) autonomous driving dataset.

**Author:** YEOH LEE MING
**Course:** Applied Machine Learning (Master in AI)
**Institution:** Asia Pacific University of Technology & Innovation (APU)
**Date:** May 2026

---

## Project overview

Modern autonomous driving systems must continuously interpret their operating context — the time of day, weather conditions, surrounding traffic density, and the presence of vulnerable road users (VRUs). While much of the published literature focuses on object detection and trajectory prediction, **scene-level operational-context classification** has received comparatively less attention as a standalone task.

This project investigates whether **classical machine learning** on **hand-crafted visual features** can reliably predict multiple operational scene attributes from a single front-camera image on nuScenes. We use only pre-deep-learning techniques (HOG, color histograms, LBP, photometric statistics) paired with five classical model families (Logistic Regression, SVM, Random Forest, XGBoost, and a shallow Multi-Layer Perceptron with one hidden layer).

### The four scene attributes

| # | Attribute | Classes | Source |
|---|---|---|---|
| 1 | `time_of_day` | day / night | Scene description |
| 2 | `weather` | clear / rain | Scene description |
| 3 | `vehicle_density` | low / medium / high | Tertile bins on forward-cone vehicle count |
| 4 | `vru_present` | absent / present | Forward-cone VRU annotation count > 0 |

### The five classical model families

| # | Model | Family | Decision boundary |
|---|---|---|---|
| 1 | Logistic Regression | Linear / probabilistic | Hyperplane |
| 2 | SVM (RBF kernel) | Kernel-based | Curved, margin-maximising |
| 3 | Random Forest | Tree ensemble (bagging) | Axis-aligned splits, averaged |
| 4 | XGBoost | Tree ensemble (boosting) | Sequentially corrected splits |
| 5 | MLP (1-2 hidden layers) | Shallow neural network | Non-linear, fully-connected |

### Research questions

- **RQ1.** Can hand-crafted visual features support multi-attribute scene classification on nuScenes front-camera images?
- **RQ2.** Which scene attributes are most and least learnable from a single front-camera frame?
- **RQ3.** Among classical model families, which performs most consistently across attributes?
- **RQ4.** Which feature types (HOG, color histograms, LBP, photometric statistics) are most informative for each attribute?
- **RQ5.** How much does hyperparameter tuning improve over base (default-setting) models?

---

## Repository structure

```
nuscenes-multi-attribute-classification/
├── data/
│   ├── nuscenes/v1.0-mini/                 # raw dataset (NOT in repo — see Setup)
│   ├── metadata/
│   │   ├── sample_metadata.csv             # per-keyframe metadata
│   │   └── sample_metadata_enriched.csv    # + forward-cone object counts
│   ├── labels/
│   │   ├── attribute_labels.csv            # one row per keyframe, 4 attributes
│   │   └── attribute_thresholds.json       # reproducibility metadata
│   ├── features/
│   │   ├── features_full.csv               # ~3,100 features per keyframe
│   │   └── feature_metadata.json           # feature-group → column-index map
│   └── splits/
│       ├── scene_split_assignments.csv     # scene_token → split mapping
│       ├── train.csv / val.csv / test.csv  # ready-to-model splits
│
├── notebooks/
│   ├── 01_eda.ipynb                  # exploratory data analysis
│   ├── 02_attribute_labels.ipynb     # generate 4 attribute labels
│   ├── 03_image_features.ipynb       # HOG + color hist + LBP + photometric
│   ├── 04_splits.ipynb               # scene-aware train/val/test split
│   ├── 05_classical_models.ipynb     # 5 models × base+tuned × 4 attrs × 3 seeds
│   ├── 06_feature_ablation.ipynb     # 5 feature subsets × 5 models × 4 attrs × 3 seeds
│   └── 07_results_analysis.ipynb     # cross-attribute analysis + statistical tests
│
├── models/                                  # 120 trained .pkl checkpoints
├── results/
│   ├── metrics/                             # all_metrics.csv, ablation_metrics.csv
│   ├── predictions/                         # per-sample test predictions
│   ├── final/                               # headline table, statistical tests, summary JSON
│   └── figures/                             # all generated plots
│
├── report/
│   └── report.pdf                           # final written report
│
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10–3.12
- ~5 GB disk space for v1.0-mini, ~350 GB for v1.0-trainval
- CPU only (classical ML — no GPU needed)
- Modern multi-core CPU recommended for grid search (8+ cores ideal)

### Installation

```bash
# Clone and enter
git clone <your-repo-url>
cd nuscenes-multi-attribute-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
.venv\Scripts\activate              # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The nuScenes dataset is not redistributed in this repository. Download separately:

1. Create a free account at [nuscenes.org](https://www.nuscenes.org/nuscenes#download).
2. Download **`v1.0-mini`** (~4 GB).
3. Extract into `data/nuscenes/v1.0-mini/` so the structure looks like:
   ```
   data/nuscenes/v1.0-mini/
   ├── maps/
   ├── samples/
   ├── sweeps/
   └── v1.0-mini/         # metadata JSONs
   ```

To later scale up, repeat with `v1.0-trainval` and change `DATASET_VERSION` in each notebook from `'v1.0-mini'` to `'v1.0-trainval'`.

---

## How to run

Execute the notebooks **in order**. Each notebook produces outputs consumed by the next.

| # | Notebook | Runtime (v1.0-mini) | Output |
|---|---|---|---|
| 1 | `01_eda.ipynb` | ~1 min | Class-distribution plots, image stats |
| 2 | `02_attribute_labels.ipynb` | ~30 sec | `attribute_labels.csv` with 4 attributes |
| 3 | `03_image_features.ipynb` | ~2-5 min | `features_full.csv` with ~3,100 features |
| 4 | `04_splits.ipynb` | ~10 sec | Scene-aware train/val/test splits |
| 5 | `05_classical_models.ipynb` | ~15-30 min | 120 trained models + headline metrics |
| 6 | `06_feature_ablation.ipynb` | ~25-60 min | 300 ablation runs across feature subsets |
| 7 | `07_results_analysis.ipynb` | ~30 sec | Final tables, figures, statistical tests |

> All notebooks are seeded for reproducibility (`SEED_LIST = [42, 7, 123]` for multi-run reliability).

---

## Methodology highlights

- **Scene-aware splitting.** Train/val/test split is performed at the *scene* level, not the frame level, to prevent leakage between adjacent keyframes from the same 20-second scene. A greedy class-balanced assignment ensures rare classes (e.g., `night`) are distributed across splits rather than landing entirely in train or test.
- **Forward-cone filter for object counts.** Vehicle and VRU counts use a forward-cone (±30°, 50 m) filter via 3D annotation projection, matching what an ADAS front camera "sees" rather than the full 360° annotation.
- **Train-only feature scaling.** `StandardScaler` is fit on the training split only; validation and test data are transformed with the same parameters, preventing test-statistic leakage.
- **Base vs Tuned comparison.** All 5 models are trained twice — once with sklearn defaults (base), once with grid-searched hyperparameters and class-weighted training (tuned). Direct comparison quantifies the value of careful model setup.
- **Multi-seed evaluation.** All results are reported as mean ± std across 3 random seeds. Paired Wilcoxon signed-rank tests assess whether observed differences (tuned vs base, model vs model) are statistically significant.
- **Feature-group ablation.** Each tuned model is retrained on five feature subsets (HOG only, color only, LBP only, photometric only, all combined) to identify which feature types are most informative per attribute.

---

## Key limitations

This project uses **v1.0-mini** (10 scenes, 404 keyframes), a development-scale split designed for prototyping. Specific consequences:

- **Small absolute sample counts**, especially for minority classes (e.g., night, rain).
- **Limited operational diversity** — only 10 scenes covering a small slice of weather and time-of-day conditions.
- **Wide confidence intervals** on per-class metrics; statistical tests have limited power with only 3 seeds × 5 models = 15 paired observations.
- **Scene-level correlations** between attributes (e.g., night and rain often co-occur in driving datasets) inflate apparent learnability.

Scaling to **v1.0-trainval** (~34,000 keyframes) is identified as future work and supported by the notebook structure (single variable change).

---

## Results summary

_To be populated after notebook 07 is complete._

| Attribute | Best Model | Test Macro-F1 | Difficulty Tier |
|---|---|---|---|
| `time_of_day` | _model_ | _x.xxx ± y.yyy_ | _easy/medium/hard_ |
| `weather` | _model_ | _x.xxx ± y.yyy_ | _tier_ |
| `vehicle_density` | _model_ | _x.xxx ± y.yyy_ | _tier_ |
| `vru_present` | _model_ | _x.xxx ± y.yyy_ | _tier_ |

**Most consistent model across attributes:** _model_ (mean macro-F1 = _x.xxx_)

---

## Reproducibility

All notebooks use seeded random number generators. Hyperparameters and feature-engineering thresholds are documented in `attribute_thresholds.json` and `feature_metadata.json`. Trained model checkpoints are saved under `models/`.

To reproduce results from scratch:

```bash
jupyter notebook notebooks/
# Run notebooks 01 through 07 in order
```

---

## References

The methodology is informed by literature spanning:
- Autonomous-driving datasets (nuScenes, KITTI, Waymo)
- Hand-crafted image features (HOG, LBP, color histograms)
- Classical machine learning (logistic regression, SVMs, tree ensembles, XGBoost, neural networks)
- Multi-attribute / multi-task classification

A non-exhaustive list of foundational works:

- Caesar, H. et al. (2020). *nuScenes: A multimodal dataset for autonomous driving.* CVPR.
- Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for human detection.* CVPR.
- Ojala, T. et al. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.* TPAMI.
- He, K. et al. (2016). *Deep residual learning for image recognition.* CVPR. _(referenced for context)_
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks.* Machine Learning.
- Breiman, L. (2001). *Random forests.* Machine Learning.

A complete list of ≥ 15 papers is included in `report/report.pdf`.

---

## License

This is an academic project. The code is provided as-is for educational purposes.
The nuScenes dataset is licensed separately by its authors; see [nuscenes.org](https://www.nuscenes.org/terms-of-use) for terms of use.

---

## Acknowledgements

- nuScenes dataset by Motional (formerly nuTonomy).
- Course staff for guidance and feedback.
