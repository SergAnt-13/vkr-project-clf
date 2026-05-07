import tempfile
import unittest
from pathlib import Path

import pandas as pd

from config import Config
from data_loader import DataLoader
from okpd_classifier import OKPDClassifier
from text_processor import TextProcessor
from vat_processor import VATProcessor


class CoreTests(unittest.TestCase):
    def test_single_column_abbreviations_file_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "abbr.xlsx"
            pd.DataFrame(
                {
                    "abbr;expansion;category;notes": [
                        "г;грамм;единицы;",
                        "мл;миллилитр;единицы;",
                    ]
                }
            ).to_excel(tmp_path, index=False)

            cfg = Config()
            cfg.ABBREVIATIONS_FILE = tmp_path
            rules = DataLoader(cfg).load_abbreviations()

            self.assertEqual(rules[0], (r"\b(?:г)\b", "грамм"))
            self.assertEqual(rules[1], (r"\b(?:мл)\b", "миллилитр"))

    def test_text_normalization_keeps_meaningful_text(self):
        processor = TextProcessor(abbreviation_rules=[("г", "грамм")], config_obj=Config())
        normalized = processor.normalize_text("МОЛОКО 25г, АКЦ.")

        self.assertIn("молоко", normalized)
        self.assertNotIn(",", normalized)

    def test_okpd_code_validation(self):
        classifier = OKPDClassifier(Config())
        self.assertTrue(classifier.is_valid_code("10.72.11.120"))
        self.assertFalse(classifier.is_valid_code("abc"))

    def test_vat_prefix_generation(self):
        processor = VATProcessor(Config())
        prefixes = processor.generate_prefixes("10.72.11.120")
        self.assertEqual(prefixes, ["10.72", "10.72.11", "10.72.11.120"])


if __name__ == "__main__":
    unittest.main()
