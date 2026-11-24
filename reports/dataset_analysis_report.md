# RetinaInsight Dataset Analysis

This report summarizes the supplied, labeled APTOS retinal images using the local validation and analysis workflow. All results below are computed from the files in `data/raw/`.

## Dataset overview

The source contains **3,662 labeled records** across three supplied partitions and **3,662 image files**. The original partition is retained in the metadata for traceability. The new manifests were generated from the complete labeled pool.

## Data validation

| Check | Result |
|:--|--:|
| Original csv records | 3662 |
| Total image files found | 3662 |
| Valid readable images | 3662 |
| Invalid or unreadable images | 0 |
| Missing image files | 0 |
| Orphan image files | 0 |
| Duplicate csv image ids | 0 |
| Invalid or missing labels | 0 |
| Exact duplicate groups | 123 |
| Conflicting duplicate groups | 30 |
| Conflicting duplicate rows | 62 |
| Validation errors recorded | 0 |
| Final modelling images | 3600 |

Every labeled row matched an image file, and all image files were readable. No invalid labels, duplicate CSV IDs, or orphan images were reported.

## Diagnosis distribution

Source labels (all records):

| Grade | Diagnosis | Images | Share |
|---:|:--|---:|---:|
| 0 | No DR | 1805 | 49.29% |
| 1 | Mild | 370 | 10.10% |
| 2 | Moderate | 999 | 27.28% |
| 3 | Severe | 193 | 5.27% |
| 4 | Proliferative DR | 295 | 8.06% |

Clean modelling candidates after unresolved label conflicts are held aside:

| Grade | Diagnosis | Images | Share |
|---:|:--|---:|---:|
| 0 | No DR | 1801 | 50.03% |
| 1 | Mild | 360 | 10.00% |
| 2 | Moderate | 976 | 27.11% |
| 3 | Severe | 180 | 5.00% |
| 4 | Proliferative DR | 283 | 7.86% |

Grade 0 is the majority category and grade 3 is the smallest. Keep the natural distribution during data exploration. During model development, consider documented imbalance methods such as class-weighted loss rather than changing the source data.

## Image properties

| Property | Minimum | Median | Mean | Maximum |
|:--|--:|--:|--:|--:|
| Width (pixels) | 474.000 | 2144.000 | 2015.177 | 4288.000 |
| Height (pixels) | 358.000 | 1536.000 | 1526.830 | 2848.000 |
| File size (bytes) | 221737.000 | 1934951.000 | 2349280.394 | 7674292.000 |
| Aspect ratio | 1.000 | 1.333 | 1.284 | 1.506 |

Image formats: PNG: 3662.

Color modes: RGB: 3662.

The generated charts are `class_distribution.png`, `class_distribution_pie.png`, `image_dimensions.png`, `image_properties.png`, and `sample_images.png`.

## Duplicate and label audit

SHA-256 found **123 exact-duplicate groups**. **30 groups (62 records)** contain identical image bytes with different diagnosis labels; **11 of those groups** span the source partitions. Their source rows and files are preserved in `image_metadata.csv` and `conflicting_duplicate_review.csv`. Because no trusted source is available to determine the correct labels, these records are excluded from `clean_metadata.csv` and the split manifests pending verification. No label was changed and no source file was deleted.

The other 93 exact-duplicate groups have consistent labels and are kept in a single split. SHA-256 detects byte-identical files; visual duplicates with different encoding or metadata may not be detected.

## Split manifests

| Split | Images | Share |
|:--|--:|--:|
| Training | 2521 | 70.03% |
| Validation | 540 | 15.00% |
| Testing | 539 | 14.97% |

The split was stratified with `random_state=42` after grouping exact hashes, so proportions are approximate by image count. The split manifests have no image-ID or exact-hash overlap. The source labels do not provide reliable patient identifiers, so patient-level independence cannot be guaranteed.

## Limits and next steps

The 62 records with conflicting labels need a trusted label source or expert adjudication before they can be used for modelling. Readability and file metadata do not establish diagnostic correctness. Later preprocessing should document resizing, normalization, and quality exclusions while preserving the source files and audit trail.

