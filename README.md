# RetinaInsight: Transfer Learning for Explainable Diabetic Retinopathy Screening

**Project ID:** SIH26038  
**Domain:** Deep Learning, Medical Computer Vision, Explainable AI (XAI), Healthcare Data Engineering  
**Dataset:** APTOS 2019 Blindness Detection  

---

## Executive Summary

**RetinaInsight** is an end-to-end, reproducible deep learning system engineered to support clinical screening for Diabetic Retinopathy (DR) from color fundus photographs. The application classifies retinal imagery into five internationally recognized severity grades:

- **Grade 0:** No DR
- **Grade 1:** Mild Diabetic Retinopathy
- **Grade 2:** Moderate Diabetic Retinopathy
- **Grade 3:** Severe Diabetic Retinopathy
- **Grade 4:** Proliferative Diabetic Retinopathy (PDR)

The platform compares three convolutional neural network architectures (**EfficientNetB0**, **ResNet50**, and **MobileNetV2**) using ImageNet transfer learning. It incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate visual saliency heatmaps highlighting retinal lesions (such as microaneurysms, hemorrhages, and exudates) that drive model predictions. The system is backed by an automated SQLite database for audit trails and analytics, and an interactive **Streamlit** user interface.

---

## System Architecture

```text
RetinaInsight/
├── configs/
│   └── training_config.py      # Centralized hyperparameter and filesystem configuration
├── data/
│   ├── raw/                    # Raw fundus photographs and CSV manifests (git-ignored)
│   ├── processed/              # Processed / cached dataset artifacts
│   ├── metadata/               # Stratified manifests and audit reports
│   └── database/               # Local SQLite database (retinainsight.db)
├── models/
│   ├── checkpoints/            # Intermediate epoch checkpoints (.keras)
│   └── saved/                  # Trained model weights (.keras)
├── src/
│   ├── data/
│   │   ├── dataset_analysis.py # Dataset integrity checks and duplicate hashing audit
│   │   └── dataset_loader.py   # tf.data pipeline with architecture-specific normalization
│   ├── models/
│   │   └── transfer_models.py  # Model factory for EfficientNetB0, ResNet50, MobileNetV2
│   ├── training/
│   │   └── train.py            # 2-stage transfer learning training orchestrator
│   ├── evaluation/
│   │   ├── metrics.py          # Clinical metrics (QWK, Macro/Weighted F1, ROC-AUC)
│   │   ├── visualizer.py       # Loss/accuracy curves and confusion matrix plots
│   │   └── evaluate.py         # Multi-model evaluation runner
│   ├── explainability/
│   │   └── gradcam.py          # Target-layer Grad-CAM heatmaps & visual overlays
│   └── database/
│       ├── schema.py           # Relational schema for predictions and evaluations
│       └── db_manager.py       # Thread-safe SQLite manager with analytics queries
├── app/
│   └── app.py                  # Multi-tab Streamlit screening & analytics dashboard
├── outputs/
│   ├── figures/                # Exported training curves and confusion matrices
│   └── reports/                # Evaluation metrics JSON and analytical summaries
├── tests/                      # Automated unit test suite
├── requirements.txt            # Environment dependencies
└── README.md
```

---

## Dataset & Quality Assurance Audit

The project uses the **APTOS 2019 Blindness Detection** dataset (3,662 labeled images). A comprehensive cryptographic and metadata audit established data integrity:

- **100% Readability:** All 3,662 image files were verified for decodability, dimensions, and channels.
- **Label Conflict Segregation:** Cryptographic SHA-256 fingerprinting detected 30 duplicate image groups (62 records) containing identical image bytes assigned contradictory severity grades. These 62 records are held out in `data/metadata/conflicting_duplicate_review.csv` to prevent label noise during model optimization.
- **Stratified Partitioning:** The remaining 3,600 clean images were split into stratified partitions with exact duplicate group isolation to prevent cross-partition leakage:
  - **Train:** 2,521 records (70.0%)
  - **Validation:** 540 records (15.0%)
  - **Test:** 539 records (15.0%)

### Class Distribution and Imbalance Strategy

| Grade | Diagnosis | Train Count | Train Share | Balanced Class Weight |
|:---:|:---|:---:|:---:|:---:|
| 0 | No DR | 1,260 | 50.0% | **0.400** |
| 1 | Mild | 250 | 9.9% | **2.017** |
| 2 | Moderate | 687 | 27.2% | **0.734** |
| 3 | Severe | 126 | 5.0% | **4.002** |
| 4 | Proliferative DR | 198 | 7.9% | **2.546** |

To resolve the 10:1 class imbalance between Grade 0 and Grade 3 without synthesizing artificial pixels, balanced class weights are computed strictly from the training partition:
$$w_j = \frac{N_{\text{total}}}{K \cdot N_j}$$

---

## Transfer Learning & Methodology

### Architecture-Specific Preprocessing
- **EfficientNetB0:** Retains $[0, 255]$ scale (internally rescaled within Keras architecture).
- **ResNet50:** Zero-centered RGB-to-BGR channel normalization using ImageNet statistics.
- **MobileNetV2:** Normalized to range $[-1.0, +1.0]$.

### Medically Safe Augmentation (Training Only)
Fundus imagery possesses radial symmetry; horizontal and vertical flips are applied alongside conservative brightness variations ($\pm 6\%$) to preserve diagnostic microaneurysms and hard exudates.

### Standardized Screening Head
To maintain an unbiased comparison, all three architectures utilize an identical classification head:
$$\text{Input (224, 224, 3)} \longrightarrow \text{Backbone} \longrightarrow \text{GlobalAvgPool2D} \longrightarrow \text{BatchNorm} \longrightarrow \text{Dense(256, ReLU)} \longrightarrow \text{Dropout(0.3)} \longrightarrow \text{Dense(5, Softmax)}$$

### Two-Stage Optimization
1. **Stage 1 (Feature Extraction):** Backbone weights frozen; the top dense head is trained with $\text{Adam}(\text{lr}=10^{-3})$.
2. **Stage 2 (Fine-Tuning):** Top 20 backbone layers unfrozen; trained with $\text{Adam}(\text{lr}=10^{-4})$ and `ReduceLROnPlateau`. Batch normalization layers remain frozen to preserve running statistics.

---

## Explainable AI (Grad-CAM)

To provide interpretable predictions, the platform implements Gradient-weighted Class Activation Mapping:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

- **EfficientNetB0 Target:** `top_activation`
- **ResNet50 Target:** `conv5_block3_out`
- **MobileNetV2 Target:** `out_relu`

The resulting 2D activation map is normalized, converted to a colormap, and blended over the fundus image at adjustable transparency.

---

## Quick Start & Execution

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/AaruhiShrivastva/Retina-Insight-Explainable-DR-Screening.git
cd "Retina-Insight-Explainable-DR-Screening"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Run Automated Verification Tests

```powershell
python -m unittest tests/test_data_pipeline.py
python -m unittest tests/test_transfer_models.py
python -m unittest tests/test_training_and_evaluation.py
python -m unittest tests/test_gradcam.py
python -m unittest tests/test_database.py
```

### 3. Train a Transfer Learning Model

```powershell
# Fast benchmark run on MobileNetV2
python -m src.training.train --architecture mobilenetv2 --epochs-stage1 1 --epochs-stage2 1 --subsample 0.05

# Full training run
python -m src.training.train --architecture mobilenetv2 --epochs-stage1 5 --epochs-stage2 10
```

### 4. Evaluate on Validation or Test Splits

```powershell
python -m src.evaluation.evaluate --architecture mobilenetv2 --split validation
```

### 5. Launch the Streamlit Screening Application

```powershell
streamlit run app/app.py
```

The web dashboard opens at `http://localhost:8501` featuring:
- **Screening & Grad-CAM:** Upload fundus images or select validation samples to view predicted severity, confidence, and visual saliency heatmaps.
- **Screening History:** Real-time query log backed by SQLite.
- **Analytics Dashboard:** Metrics on total screenings, severity distribution, and model utilization.
- **Model Comparison:** Performance tables and confusion matrices.
- **Ethics & Limitations:** Medical disclaimers and clinical boundary documentation.

---

## Scientific & Ethical Limitations

1. **Research Screening Tool:** RetinaInsight is an academic screening-support tool, not an FDA-cleared diagnostic device. It is intended to assist triage, not replace clinical diagnosis.
2. **Clinical Adjudication:** All machine predictions must be verified by a board-certified ophthalmologist or optometrist before clinical intervention.
3. **Generalization:** Models are trained on the APTOS 2019 dataset. External validation across diverse imaging systems and patient populations is necessary prior to clinical deployment.
4. **Saliency Interpretation:** Grad-CAM heatmaps represent computational activation correlations and do not constitute direct biological or pathological proof.

---

## Team

- **Aaruhi** — 12311428 — Roll No. 49
- **Atul Mor** — 12323724 — Roll No. 31
- **Priya Yadav** — 12321119 — Roll No. 28
