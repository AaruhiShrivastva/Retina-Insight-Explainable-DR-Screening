import hashlib
import tempfile
import unittest
from pathlib import Path

from src.data.dataset_analysis import CLASS_NAMES, sha256_file


class DatasetAnalysisUtilityTests(unittest.TestCase):
    def test_sha256_is_repeatable_for_identical_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.bin"
            second = Path(folder) / "second.bin"
            first.write_bytes(b"retina insight")
            second.write_bytes(b"retina insight")
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(sha256_file(first), hashlib.sha256(b"retina insight").hexdigest())

    def test_diagnosis_labels_cover_five_classes(self):
        self.assertEqual(set(CLASS_NAMES), {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()

