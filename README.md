# nuscenes-scene-classification-ml

A comparative study of classical machine-learning models for multi-attribute scene classification of front-camera driving images on the [nuScenes](https://www.nuscenes.org/) autonomous driving dataset.

| Field | Details |
|---|---|
| **Author** | YEOH LEE MING |
| **Course** | Applied Machine Learning (Master in AI) |
| **Institution** | Asia Pacific University of Technology & Innovation (APU) |
| **Date** | May 2026 |

---

## Project overview

Modern autonomous driving systems must continuously interpret their operating context — the time of day, weather conditions, surrounding traffic density, and the presence of vulnerable road users (VRUs). While much of the published literature focuses on object detection and trajectory prediction, **scene-level operational-context classification** has received comparatively less attention as a standalone task.

This project investigates whether **classical machine learning** on **hand-crafted visual features** can reliably predict multiple operational scene attributes from a single front-camera image on nuScenes. We use only pre-deep-learning techniques (HOG, color histograms, LBP, photometric statistics) paired with five classical model families, evaluated using **5-fold scene-aware cross-validation** following Demšar (2006).

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
| 5 | MLP (1 hidden layer) | Shallow neural network | Non-linear, fully-connected |

### Research questions

- **RQ1.** Can hand-crafted visual features support multi-attribute scene classification on nuScenes front-camera images?
- **RQ2.** Which scene attributes are most and least learnable from a single front-camera frame?
- **RQ3.** Among classical model families, which performs most consistently across attributes?
- **RQ4.** Which feature types (HOG, color histograms, LBP, photometric statistics) are most informative for each attribute?
- **RQ5.** How much does hyperparameter tuning and class-weighted training improve over base classifiers?

---

## Key results (v1.0-mini, 5-fold scene-aware CV)

| Attribute | Best Model | Mean Macro-F1 | Difficulty Tier |
|---|---|---|---|
| `time_of_day` | XGBoost | 0.968 | Easy |
| `weather` | SVM | 0.870 | Easy |
| `vehicle_density` | LogReg | 0.278 | Hard |
| `vru_present` | MLP | 0.437 | Hard |

**Most consistent model across attributes:** XGBoost

**Key finding:** Hand-crafted features support photometric attribute classification (time_of_day, weather) but are insufficient for shape-based counting tasks (vehicle_density, vru_present) at v1.0-mini scale. The asymmetry reflects the discriminative axis of each attribute — global lighting statistics are captured well by color/photometric features; object counting requires spatial reasoning beyond classical descriptors.

---

## Repository structure

```
nuscenes-scene-classification-ml/
├── data/
│   ├── nuscenes/v1.0-mini/                 # raw dataset (NOT in repo — see Setup)
│   ├── metadata/
│   │   ├── sample_metadata.csv             # per-keyframe metadata (notebook 01)
│   │   └── sample_metadata_enriched.csv    # + forward-cone object counts
│   ├── labels/
│   │   ├── attribute_labels.csv            # one row per keyframe, 4 attributes
│   │   └── attribute_thresholds.json       # label-generation reproducibility metadata
│   ├── features/
│   │   ├── features_full.csv               # ~6,200 features per keyframe
│   │   ├── feature_metadata.json           # feature-group → column-index map
│   │   ├── features_pca.csv                # PCA-projected features (fold 0)
│   │   └── features_lda_<attr>.csv         # LDA-projected features per attribute
│   └── splits/
│       ├── fold_assignments.csv            # scene_name → fold mapping
│       ├── fold_0_train.csv                # k-fold train/test CSVs (5 folds)
│       ├── fold_0_test.csv
│       ├── ... (fold_1 through fold_4)
│       └── kfold_metadata.json             # fold structure metadata
│
├── notebooks/
│   ├── 01_eda.ipynb                        # exploratory data analysis
│   ├── 02_attribute_labels.ipynb           # generate 4 attribute labels
│   ├── 03_image_features.ipynb             # HOG + color hist + LBP + photometric
│   │                                        # + preprocessing doc + synthetic demos
│   ├── 04_splits.ipynb                     # 5-fold scene-aware CV splits
│   ├── 05a_dim_reduction.ipynb             # PCA + LDA (fold 0 representative)
│   ├── 05b_preprocessing_ablation.ipynb    # hist_eq vs CLAHE vs none (fold 0)
│   ├── 06_classical_models.ipynb           # 5 models × base+tuned × 4 attrs
│   │                                        # × 3 seeds × 5 folds = 600 fits
│   ├── 07_feature_ablation.ipynb           # 5 subsets × 5 models × 4 attrs
│   │                                        # × 3 seeds × 5 folds = 1500 fits
│   ├── 08a_headline_results.ipynb          # k-fold aggregated tables
│   ├── 08b_visual_analysis_and_stats.ipynb # confusion matrices, ROC, Wilcoxon tests
│   └── 08c_deep_analysis.ipynb             # feature importance, error analysis,
│                                            # RQ summary + scaling roadmap
│
├── models/
│   └── fold_<k>/<attr>/                    # trained .pkl checkpoints per fold
│
├── results/
│   ├── metrics/
│   │   ├── all_metrics.csv                 # 570 test-split rows (k-fold)
│   │   ├── baseline_metrics.csv            # random + majority-class baselines
│   │   ├── skipped_combos.json             # (fold, attr) combos skipped
│   │   ├── ablation_metrics.csv            # 1425 feature ablation rows
│   │   ├── aggregated_metrics.csv          # mean ± std across folds × seeds
│   │   ├── dim_reduction_metrics.csv       # PCA/LDA comparison
│   │   └── preprocessing_ablation_metrics.csv
│   ├── predictions/
│   │   └── predictions_test.csv            # 44,910 per-sample predictions
│   ├── final/
│   │   ├── headline_table.csv
│   │   ├── all_results_matrix.csv
│   │   ├── top3_per_attribute.csv
│   │   ├── recall_precision_best.csv
│   │   ├── statistical_tests.csv
│   │   ├── tuning_significance.csv
│   │   ├── feature_importance_rf.csv
│   │   ├── feature_importance_lr.csv
│   │   ├── error_analysis.csv
│   │   ├── final_summary.json
│   │   └── scaling_roadmap.json            # exact steps to scale to v1.0-trainval
│   └── figures/                            # all generated plots
│
├── report/
│   └── report.pdf                          # final written report
│
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10–3.12 (tested on Python 3.12.10)
- ~5 GB disk space for v1.0-mini, ~35 GB for v1.0-trainval (camera-only)
- CPU only — no GPU needed for classical ML
- Modern multi-core CPU recommended (8+ cores ideal for grid search)

### Installation

```bash
# Clone and enter
git clone https://github.com/LeeMing988/nuscenes-scene-classification-ml.git
cd nuscenes-scene-classification-ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
.venv\Scripts\activate              # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The nuScenes dataset is **not redistributed** in this repository. Download separately:

1. Create a free account at [nuscenes.org](https://www.nuscenes.org/nuscenes#download)
2. Download **`v1.0-mini`** (~4 GB)
3. Extract into `data/nuscenes/v1.0-mini/` so the structure looks like:

```
data/nuscenes/v1.0-mini/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/         # metadata JSONs
```

---

## How to run

Execute notebooks **in this exact order**. Each notebook produces outputs consumed by the next.

| # | Notebook | Runtime (v1.0-mini) | Key output |
|---|---|---|---|
| 01 | `01_eda.ipynb` | ~1 min | Class-distribution plots |
| 02 | `02_attribute_labels.ipynb` | ~30 sec | `attribute_labels.csv` |
| 03 | `03_image_features.ipynb` | ~5-10 min | `features_full.csv` (~6,200 dims) |
| 04 | `04_splits.ipynb` | ~10 sec | `fold_0_train.csv` … `fold_4_test.csv` |
| 05a | `05a_dim_reduction.ipynb` | ~1-2 min | PCA/LDA analysis (fold 0) |
| 05b | `05b_preprocessing_ablation.ipynb` | ~5-8 min | Preprocessing comparison (fold 0) |
| 06 | `06_classical_models.ipynb` | **~1.5-3 hours** | 600 model fits, all metrics |
| 07 | `07_feature_ablation.ipynb` | **~1-2 hours** | 1500 ablation fits |
| 08a | `08a_headline_results.ipynb` | ~30 sec | Headline tables (Table 7/8/9) |
| 08b | `08b_visual_analysis_and_stats.ipynb` | ~1 min | Confusion matrices, ROC, stat tests |
| 08c | `08c_deep_analysis.ipynb` | ~1 min | Feature importance, error analysis, RQ summary |

> **Total pipeline runtime:** ~3-5 hours on a modern multi-core CPU (mostly notebooks 06 and 07).

> **Seeded for reproducibility:** `SEED_LIST = [42, 7, 123]`, `SPLIT_SEED = 42`.

---

## Methodology highlights

### 5-Fold Scene-Aware Cross-Validation (key methodological choice)

On v1.0-mini (10 scenes, 404 keyframes), a single train/val/test split yields a test set of only 1-2 scenes (~80 keyframes). With such a small test set, minority classes (rain, high vehicle density) can be entirely absent from the test split — making macro-F1 meaningless for those classes.

Following Demšar (2006), we adopt **5-fold scene-aware cross-validation**:
- Every scene appears in test exactly once across the 5 folds
- All 404 keyframes are evaluated; no scene leakage (verified explicitly)
- Per-fold metrics aggregate to mean ± std across folds × 3 seeds = **15 observations per (model, attribute)**
- Some (fold, attribute) combinations are skipped if the training set has only one class for that attribute (documented in `skipped_combos.json`)

### Other methodological choices

- **Scene-aware partitioning:** adjacent keyframes from the same scene are kept together in train or test, preventing near-duplicate leakage
- **Forward-cone filter:** vehicle and VRU counts use a ±30°, 50 m cone matching ADAS front-camera field of view
- **Train-only scaling:** `StandardScaler` fitted on training split only — no test leakage
- **Trivial baselines:** random and majority-class baselines anchored to every model comparison
- **Base vs Tuned:** all 5 models trained with sklearn defaults (base) and grid-searched + class-weighted (tuned)
- **Defensive grid-search fallback:** if a class has too few samples for inner 3-fold CV, the model falls back to base estimator (documented per run)
- **Statistical tests:** paired Wilcoxon signed-rank tests with Cohen's d effect sizes

---

## Experimental scale

| Component | Count |
|---|---|
| Model fits (notebook 06) | 570 completed (600 expected, 30 skipped) |
| Baseline computations | 114 |
| Feature ablation fits (notebook 07) | 1,425 completed (1,500 expected, 75 skipped) |
| Individual test predictions saved | 44,910 |
| Scene leakage detected | 0 |

---

## Key limitations

- **v1.0-mini scale:** only 10 scenes limits per-class statistical power and class coverage per fold
- **Rain concentration:** all rain frames are in 1-2 scenes; fold where rain scene is in test has no rain in train (documented, handled by skip-and-document)
- **Single-frame restriction:** no temporal context across frames
- **Classical features:** HOG/LBP/color histograms cannot perform object counting — explains low vehicle_density and vru_present performance
- **Tuning bundled with class-weighting:** mean tuning gain = −0.019 overall; class-weighting overcorrection on imbalanced attributes is the likely cause

---

## Scaling to v1.0-trainval

The pipeline is designed for a one-variable scale-up. See `results/final/scaling_roadmap.json` for the complete parameter-change manifest. Summary:

| Notebook | Change |
|---|---|
| 01, 02, 03 | `DATASET_VERSION = 'v1.0-trainval'` |
| 04 | Switch to single 70/15/15 stratified split (k-fold optional at scale) |
| 06 | Reduce grid search complexity (RandomizedSearchCV or smaller grids) |
| 07 | Reduce ablation matrix (2 best models only) |
| 08a–08c | Aggregation over seeds only (no fold dimension) |

Expected improvements: weather and vehicle_density become reliably evaluable; all class distributions stabilise; confidence intervals tighten significantly.

---

## Reproducibility

| Item | Where |
|---|---|
| Random seeds | `SEED_LIST = [42, 7, 123]`, `SPLIT_SEED = 42` in each notebook |
| Hyperparameters | `attribute_thresholds.json`, `feature_metadata.json` |
| Fold assignments | `data/splits/kfold_metadata.json` |
| Trained models | `models/fold_<k>/<attr>/` |
| Skipped combos | `results/metrics/skipped_combos.json` |
| Library versions | `requirements.txt` |

```bash
# Full reproduction from scratch:
jupyter notebook
# Run notebooks 01 → 02 → 03 → 04 → 05a → 05b → 06 → 07 → 08a → 08b → 08c
```

---

## References

- Caesar, H. et al. (2020). *nuScenes: A multimodal dataset for autonomous driving.* CVPR.
- Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for human detection.* CVPR.
- Ojala, T. et al. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.* TPAMI.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks.* Machine Learning.
- Breiman, L. (2001). *Random forests.* Machine Learning.
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets.* JMLR, 7, 1–30.
- Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.* Philosophical Magazine.
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics.
- Swain, M. J. & Ballard, D. H. (1991). *Color indexing.* IJCV.
- Lowe, D. G. (2004). *Distinctive image features from scale-invariant keypoints.* IJCV.
- Geiger, A. et al. (2012). *Are we ready for autonomous driving? The KITTI vision benchmark suite.* CVPR.
- Sun, P. et al. (2020). *Scalability in perception for autonomous driving: Waymo Open Dataset.* CVPR.
- Yu, F. et al. (2020). *BDD100K: A diverse driving dataset for heterogeneous multitask learning.* CVPR.
- Sakaridis, C., Dai, D. & Van Gool, L. (2018). *Semantic foggy scene understanding with synthetic data.* IJCV.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR.
- Pinggera, P. et al. (2016). *Lost and Found: Detecting small road hazards for self-driving vehicles.* IROS.

A complete annotated reading list is included in `report/report.pdf`.

---

## License

This is an academic project. Code is provided as-is for educational purposes.
The nuScenes dataset is licensed separately — see [nuscenes.org/terms-of-use](https://www.nuscenes.org/terms-of-use).

---

## Acknowledgements

- nuScenes dataset by Motional (formerly nuTonomy).
- Course staff at APU for guidance and feedback.
