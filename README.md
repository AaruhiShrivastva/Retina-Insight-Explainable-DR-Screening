# RetinaInsight

**Explainable Diabetic Retinopathy Screening System**  
Project ID: SIH26038

RetinaInsight is a reproducible retinal-image data analysis project. This repository contains the dataset validation, exploratory analysis, metadata generation, and duplicate-aware dataset splitting workflow. It is organized for direct use in Visual Studio Code, with a single Python entry point and a ready-to-use debugger configuration.

## Open and run in VS Code

1. Open the folder `Diabetic Retinol Screening` in VS Code.
2. Install the Microsoft Python extension.
3. Open **Terminal → New Terminal** and create a local environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or invoke `.venv\Scripts\python.exe` directly.
4. Run the complete analysis from the project root:

   ```powershell
   python run_analysis.py
   ```

   You can also open `run_analysis.py` and use VS Code's **Run Python File** command. The included `.vscode` configuration points VS Code to the local environment and provides a debugger launch profile.

The first run checks the CSV labels and image files, then writes metadata, charts, split files, and a report. The SHA-256 pass reads each image file and may take several minutes. Charts are saved under `reports/` rather than displayed as notebook output.

## Dataset layout

The supplied Kaggle copy is stored under `data/raw/`:

```text
data/raw/
├── train_1.csv
├── valid.csv
├── test.csv
├── train_images/train_images/*.png
├── val_images/val_images/*.png
└── test_images/test_images/*.png
```

Each CSV contains `id_code` and `diagnosis`. The analysis records the original partition in metadata, validates the labeled rows and matching image files, then creates a new reproducible 70/15/15 stratified split with seed 42. Exact duplicate images are kept in a single split. No patient identifier is available for patient-level splitting. The downloaded copy's `test.csv` is labeled; the unlabeled test set from the original competition is not included or used.

Large source images and generated metadata are excluded from Git. Keep the original image files intact; do not commit or redistribute them. The Kaggle competition rules limit use to academic, educational, or other non-commercial purposes and restrict sharing of competition data.

## Generated outputs

Running the analysis creates:

- `data/metadata/image_metadata.csv` — validation details and image properties.
- `data/metadata/clean_metadata.csv` — readable, labeled records suitable for the next modelling stage.
- `data/metadata/class_distribution.csv` — diagnosis counts among clean modelling candidates.
- `data/metadata/source_class_distribution.csv` — diagnosis counts across all labeled source rows.
- `data/metadata/conflicting_duplicate_review.csv` — preserved records with conflicting labels for the same exact image.
- `data/metadata/train_metadata.csv`, `validation_metadata.csv`, `test_metadata.csv` — stratified split manifests.
- `data/metadata/data_quality_report.csv` — computed data-quality summary.
- `reports/class_distribution.png`, `class_distribution_pie.png`, `image_dimensions.png`, `image_properties.png`, `sample_images.png` — visual summaries.
- `reports/dataset_analysis_report.md` — a report generated from the files actually present on your computer.

## Project layout

```text
Diabetic Retinol Screening/
├── .vscode/                 # VS Code interpreter and debugger settings
├── data/
│   ├── raw/                 # Supplied labeled images and CSV files; ignored by Git
│   ├── processed/
│   └── metadata/            # Generated CSV outputs; ignored by Git
├── reports/                 # Generated charts and report
├── src/data/dataset_analysis.py       # Reusable validation and analysis functions
├── tests/test_dataset_analysis.py     # Basic utility checks
├── requirements.txt
├── run_analysis.py          # Single analysis entry point
└── README.md
```

## Team

- Aaruhi — 12311428 — Roll No. 49
- Atul Mor — 12323724 — Roll No. 31
- Priya Yadav — 12321119 — Roll No. 28


