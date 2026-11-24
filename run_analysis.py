from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_analysis import run_dataset_analysis  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_dataset_analysis(PROJECT_ROOT)
    print("\nAnalysis complete. Summary:")
    for metric, value in results["summary"].items():
        print(f"- {metric.replace('_', ' ').capitalize()}: {value}")
    print(f"\nMetadata: {PROJECT_ROOT / 'data' / 'metadata'}")
    print(f"Reports:  {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
