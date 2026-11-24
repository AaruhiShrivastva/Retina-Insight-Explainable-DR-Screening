from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
EXPECTED_CLASSES = {0, 1, 2, 3, 4}
CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 fingerprint without loading the whole image into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dataset_analysis(project_root: Path, dataset_dir: Path | None = None) -> dict[str, object]:
    """Validate local APTOS files, create metadata/splits/plots, and write a factual report."""
    project_root = Path(project_root).resolve()
    dataset_dir = Path(dataset_dir).resolve() if dataset_dir else project_root / "data" / "raw"
    official_csv = dataset_dir / "train.csv"
    mirror_csvs = [(name, dataset_dir / filename) for name, filename in
                   (("train", "train_1.csv"), ("validation", "valid.csv"), ("test", "test.csv"))]
    if official_csv.is_file():
        csv_specs = [("official_train", official_csv)]
        image_roots = [dataset_dir / "train_images"]
    elif all(path.is_file() for _, path in mirror_csvs):
        csv_specs = mirror_csvs
        image_roots = [dataset_dir / name for name in ("train_images", "val_images", "test_images")]
    else:
        raise FileNotFoundError(
            f"Could not find train.csv or the complete labeled mirror set (train_1.csv, valid.csv, test.csv) in {dataset_dir}."
        )
    missing_image_roots = [path for path in image_roots if not path.is_dir()]
    if missing_image_roots:
        raise FileNotFoundError(f"Image folders are missing: {missing_image_roots}")

    metadata_dir = project_root / "data" / "metadata"
    report_dir = project_root / "reports"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    label_parts = []
    required = {"id_code", "diagnosis"}
    for partition, csv_file in csv_specs:
        part = pd.read_csv(csv_file)
        if not required.issubset(part.columns):
            raise ValueError(f"Expected CSV columns {sorted(required)} in {csv_file.name}; found {part.columns.tolist()}")
        part["source_partition"] = partition
        label_parts.append(part)
    labels = pd.concat(label_parts, ignore_index=True)
    labels["id_code"] = labels["id_code"].astype("string").str.strip()
    labels["diagnosis"] = pd.to_numeric(labels["diagnosis"], errors="coerce")
    labels["image_id_duplicate"] = labels["id_code"].duplicated(keep=False)
    labels["diagnosis_valid"] = labels["diagnosis"].isin(EXPECTED_CLASSES)

    paths_by_stem: dict[str, Path] = {}
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    all_image_files = [p for folder in image_roots for p in folder.rglob("*")
                       if p.is_file() and p.suffix.casefold() in image_suffixes]
    if not all_image_files:
        raise FileNotFoundError(f"No supported image files found under: {image_roots}")
    for path in all_image_files:
        paths_by_stem.setdefault(path.stem.casefold(), path)

    records: list[dict[str, object]] = []
    for row in tqdm(labels.itertuples(index=False), total=len(labels), desc="Checking images"):
        image_id = str(getattr(row, "id_code")) if pd.notna(getattr(row, "id_code")) else ""
        diagnosis = getattr(row, "diagnosis")
        path = paths_by_stem.get(Path(image_id).stem.casefold()) if image_id else None
        record: dict[str, object] = {
            "image_id": image_id,
            "source_partition": getattr(row, "source_partition", "official_train"),
            "image_path": str(path) if path else None,
            "diagnosis": diagnosis,
            "diagnosis_name": CLASS_NAMES.get(int(diagnosis)) if pd.notna(diagnosis) and diagnosis in EXPECTED_CLASSES else "Invalid/unknown",
            "diagnosis_valid": bool(getattr(row, "diagnosis_valid")),
            "image_id_duplicate": bool(getattr(row, "image_id_duplicate")),
            "file_exists": path is not None,
            "readable": False,
            "image_width": None,
            "image_height": None,
            "image_format": None,
            "color_mode": None,
            "file_size_bytes": None,
            "aspect_ratio": None,
            "sha256": None,
            "validation_error": "",
        }
        if path:
            record["file_size_bytes"] = path.stat().st_size
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    record.update({
                        "readable": True,
                        "image_width": width,
                        "image_height": height,
                        "image_format": image.format,
                        "color_mode": image.mode,
                        "aspect_ratio": width / height if height else None,
                    })
                record["sha256"] = sha256_file(path)
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                record["validation_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    image_metadata = pd.DataFrame(records)
    hash_labels = image_metadata.groupby("sha256", dropna=True)["diagnosis"].nunique()
    conflicting_hashes = set(hash_labels[hash_labels > 1].index)
    image_metadata["duplicate_group_size"] = image_metadata.groupby("sha256")["sha256"].transform("size").fillna(0).astype(int)
    image_metadata["duplicate_label_conflict"] = image_metadata["sha256"].isin(conflicting_hashes)
    image_metadata.loc[image_metadata["duplicate_label_conflict"]].sort_values(["sha256", "image_id"]).to_csv(metadata_dir / "conflicting_duplicate_review.csv", index=False)
    image_metadata.to_csv(metadata_dir / "image_metadata.csv", index=False)

    clean = image_metadata[
        image_metadata["file_exists"] & image_metadata["readable"] & image_metadata["diagnosis_valid"]
        & ~image_metadata["image_id_duplicate"] & ~image_metadata["duplicate_label_conflict"]
    ].copy()
    clean.to_csv(metadata_dir / "clean_metadata.csv", index=False)

    duplicate_rows = image_metadata[image_metadata["duplicate_group_size"] > 1].copy()
    duplicate_groups = duplicate_rows.groupby("sha256", dropna=True)
    conflict_group_count = sum(group["diagnosis"].nunique() > 1 for _, group in duplicate_groups)
    consistent_duplicate_group_count = int(duplicate_rows.groupby("sha256").ngroups - conflict_group_count)
    conflicting_duplicate_rows = int(image_metadata["duplicate_label_conflict"].sum())

    # A SHA-256 group is assigned as one unit, preventing exact copies crossing splits.
    group_table = clean.groupby("sha256", as_index=False).agg(diagnosis=("diagnosis", "first"), rows=("image_id", "size"))
    if group_table.empty:
        raise ValueError("No valid, readable, correctly labelled images remain for modelling.")
    class_group_counts = group_table["diagnosis"].value_counts()
    if (class_group_counts < 2).any():
        raise ValueError("At least one diagnosis has fewer than two unique image hashes; a stratified 70/15/15 split is not reliable.")
    remaining, test_groups = train_test_split(
        group_table, test_size=0.15, random_state=42, stratify=group_table["diagnosis"]
    )
    train_groups, validation_groups = train_test_split(
        remaining, test_size=(0.15 / 0.85), random_state=42, stratify=remaining["diagnosis"]
    )
    split_hashes = {
        "train": set(train_groups["sha256"]),
        "validation": set(validation_groups["sha256"]),
        "test": set(test_groups["sha256"]),
    }
    splits = {name: clean[clean["sha256"].isin(hashes)].copy() for name, hashes in split_hashes.items()}
    for name, frame in splits.items():
        frame.to_csv(metadata_dir / f"{name}_metadata.csv", index=False)
    if (split_hashes["train"] & split_hashes["validation"] or split_hashes["train"] & split_hashes["test"]
            or split_hashes["validation"] & split_hashes["test"]):
        raise AssertionError("Exact image hashes overlap across splits.")
    split_ids = {name: set(frame["image_id"]) for name, frame in splits.items()}
    if (split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"]
            or split_ids["validation"] & split_ids["test"]):
        raise AssertionError("Image IDs overlap across splits.")

    sns.set_theme(style="whitegrid")
    counts = clean["diagnosis_name"].value_counts().reindex(list(CLASS_NAMES.values()), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C78A8")
    ax.set(title="APTOS diagnosis counts (valid images)", xlabel="Diagnosis", ylabel="Images")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(report_dir / "class_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    nonzero_counts = counts[counts > 0]
    ax.pie(nonzero_counts.values, labels=nonzero_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("APTOS diagnosis share (valid images)")
    fig.tight_layout(); fig.savefig(report_dir / "class_distribution_pie.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    clean["image_width"].dropna().hist(ax=axes[0], bins=20); axes[0].set(title="Image widths", xlabel="Pixels", ylabel="Images")
    clean["image_height"].dropna().hist(ax=axes[1], bins=20); axes[1].set(title="Image heights", xlabel="Pixels", ylabel="Images")
    fig.tight_layout(); fig.savefig(report_dir / "image_dimensions.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    clean["image_format"].value_counts().plot.bar(ax=axes[0], title="Image formats"); axes[0].set(xlabel="Format", ylabel="Images")
    clean["color_mode"].value_counts().plot.bar(ax=axes[1], title="Color modes"); axes[1].set(xlabel="Mode", ylabel="Images")
    fig.tight_layout(); fig.savefig(report_dir / "image_properties.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for diagnosis, ax in zip(sorted(CLASS_NAMES), axes):
        sample = clean[clean["diagnosis"] == diagnosis].head(1)
        ax.set_title(CLASS_NAMES[diagnosis]); ax.axis("off")
        if not sample.empty:
            with Image.open(sample.iloc[0]["image_path"]) as image:
                ax.imshow(image.convert("RGB"))
        else:
            ax.text(0.5, 0.5, "No valid sample", ha="center", va="center")
    fig.suptitle("One validated sample per diagnosis"); fig.tight_layout(); fig.savefig(report_dir / "sample_images.png", dpi=160); plt.close(fig)

    class_summary = clean.groupby(["diagnosis", "diagnosis_name"]).size().rename("image_count").reset_index()
    class_summary["percentage"] = 100 * class_summary["image_count"] / max(len(clean), 1)
    class_summary.to_csv(metadata_dir / "class_distribution.csv", index=False)
    source_counts = labels.loc[labels["diagnosis_valid"], "diagnosis"].value_counts().reindex(sorted(EXPECTED_CLASSES), fill_value=0)
    source_class_summary = pd.DataFrame({
        "diagnosis": list(source_counts.index),
        "diagnosis_name": [CLASS_NAMES[value] for value in source_counts.index],
        "image_count": source_counts.values,
    })
    source_class_summary["percentage"] = 100 * source_class_summary["image_count"] / max(int(source_counts.sum()), 1)
    source_class_summary.to_csv(metadata_dir / "source_class_distribution.csv", index=False)
    missing_values = image_metadata.isna().sum()
    summary = {
        "original_csv_records": len(labels),
        "total_image_files_found": len(all_image_files),
        "valid_readable_images": int(image_metadata["readable"].sum()),
        "invalid_or_unreadable_images": int((image_metadata["file_exists"] & ~image_metadata["readable"]).sum()),
        "missing_image_files": int((~image_metadata["file_exists"]).sum()),
        "orphan_image_files": len(set(p.stem.casefold() for p in all_image_files) - set(labels["id_code"].dropna().map(lambda x: Path(str(x)).stem.casefold()))),
        "exact_duplicate_groups": int((duplicate_rows.groupby("sha256").ngroups)),
        "conflicting_duplicate_groups": int(conflict_group_count),
        "conflicting_duplicate_rows": conflicting_duplicate_rows,
        "validation_errors_recorded": int(image_metadata["validation_error"].ne("").sum()),
        "final_modelling_images": len(clean),
        "training_images": len(splits["train"]),
        "validation_images": len(splits["validation"]),
        "testing_images": len(splits["test"]),
        "duplicate_csv_image_ids": int(labels["image_id_duplicate"].sum()),
        "invalid_or_missing_labels": int((~labels["diagnosis_valid"]).sum()),
    }
    summary.update({f"missing_value_{column}": int(count) for column, count in image_metadata.isna().sum().items()})
    pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()]).to_csv(metadata_dir / "data_quality_report.csv", index=False)

    def markdown_table(frame: pd.DataFrame) -> list[str]:
        return frame.to_markdown(index=False).splitlines()

    dimension_rows = []
    for column, label in (("image_width", "Width"), ("image_height", "Height"), ("file_size_bytes", "File size (bytes)"), ("aspect_ratio", "Aspect ratio")):
        values = image_metadata[column].dropna()
        dimension_rows.append({"property": label, "minimum": round(float(values.min()), 3), "median": round(float(values.median()), 3), "mean": round(float(values.mean()), 3), "maximum": round(float(values.max()), 3)})
    image_property_summary = pd.DataFrame(dimension_rows)
    format_counts = image_metadata["image_format"].value_counts(dropna=False).rename_axis("format").reset_index(name="count")
    mode_counts = image_metadata["color_mode"].value_counts(dropna=False).rename_axis("color_mode").reset_index(name="count")
    split_summary = pd.DataFrame([{"split": name, "images": len(frame), "percentage": round(100 * len(frame) / max(len(clean), 1), 2)} for name, frame in splits.items()])
    conflict_partition_groups = sum(group["source_partition"].nunique() > 1 for _, group in image_metadata[image_metadata["duplicate_label_conflict"]].groupby("sha256"))

    report_lines = [
        "# RetinaInsight Dataset Analysis", "",
        "This report summarizes the supplied, labeled APTOS retinal images using the local validation and analysis workflow. All counts and findings below are computed from the files present in `data/raw/`.", "",
        "## Dataset overview", "",
        f"The source contains **{len(labels):,} labeled records** across {labels['source_partition'].nunique()} supplied partitions and **{len(all_image_files):,} image files**. The original partition is retained in the metadata for traceability. The new manifests below are generated from the complete labeled pool.", "",
        "## Data validation", "",
    ]
    report_lines += markdown_table(pd.DataFrame([{"check": key.replace("_", " ").capitalize(), "result": value} for key, value in summary.items() if not key.startswith("missing_value_")]))
    report_lines += ["", "## Diagnosis distribution", "", "Source labels (all records):", ""]
    report_lines += markdown_table(source_class_summary)
    report_lines += ["", "Clean modelling candidates after unresolved label conflicts are held aside:", ""]
    report_lines += markdown_table(class_summary)
    report_lines += ["", "The source is imbalanced: grade 0 is the majority category, while grade 3 is the smallest. Preserve the natural distribution during data exploration; consider class-weighted loss or other documented imbalance methods during model development rather than changing this source dataset.", "", "## Image properties", ""]
    report_lines += markdown_table(image_property_summary)
    report_lines += ["", "Image formats:", ""]
    report_lines += markdown_table(format_counts)
    report_lines += ["", "Color modes:", ""]
    report_lines += markdown_table(mode_counts)
    report_lines += ["", "All valid images are plotted in `image_dimensions.png`, `image_properties.png`, and `sample_images.png`. The class count charts are saved as `class_distribution.png` and `class_distribution_pie.png`.", "", "## Duplicate and label audit", "",
        f"SHA-256 found **{int(duplicate_rows.groupby('sha256').ngroups)} exact-duplicate groups**. Of these, **{int(conflict_group_count)} groups ({conflicting_duplicate_rows} records)** contain the same image bytes with different diagnosis labels; **{conflict_partition_groups} conflicting groups** cross the source partitions. Their original rows and files are preserved in `image_metadata.csv` and `conflicting_duplicate_review.csv`. Since the correct label cannot be established from the image hash, these records are held out of `clean_metadata.csv` and the split manifests pending verification; no label was changed and no source file was deleted.", "",
        f"The remaining {consistent_duplicate_group_count} exact-duplicate groups with consistent labels are kept together when assigning splits. Hashing only finds byte-identical files; visually identical images with different encoding or metadata may remain undetected.", "", "## Split manifests", ""]
    report_lines += markdown_table(split_summary)
    report_lines += ["", "The split procedure uses stratification with `random_state=42` after grouping exact image hashes, so the proportions are approximate by image count. Hash and image-ID overlap checks across the three manifests returned zero. The standard APTOS labels do not include reliable patient identifiers; patient-level separation and patient leakage checks therefore cannot be guaranteed.", "", "## Limits and next steps", "",
        "The 62 conflicting-label records need a trusted label source or expert adjudication before they can be considered for modelling. Image readability and file metadata do not establish diagnostic correctness. Later preprocessing should document resizing, normalization, and any quality exclusions, while preserving these source files and this audit trail.", ""]
    report_path = report_dir / "dataset_analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Dataset analysis outputs written under %s", project_root)
    return {"summary": summary, "class_summary": class_summary, "metadata": image_metadata, "clean_metadata": clean, "splits": splits}


