"""
Конфигурация проекта классификации ОКПД2.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Config:
    """Основная конфигурация приложения."""

    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Базовые директории проекта
    DATA_DIR: Path = field(init=False)
    TO_PROCESS_DIR: Path = field(init=False)
    TRAINING_DIR: Path = field(init=False)
    REFERENCE_DIR: Path = field(init=False)
    RESULTS_DIR: Path = field(init=False)
    ARCHIVE_DIR: Path = field(init=False)

    # Входные файлы по умолчанию
    PRODUCTS_FILE: Path = field(init=False)
    MERGED_PRODUCTS_FILE: Path = field(init=False)
    ABBREVIATIONS_FILE: Path = field(init=False)
    PP908_FILE: Path = field(init=False)
    VAT_REFERENCE_FILE: Path = field(init=False)

    # Артефакты и выходы
    OUTPUT_DIR: Path = field(init=False)
    BASELINE_OUTPUT_DIR: Path = field(init=False)
    BERT_OUTPUT_DIR: Path = field(init=False)
    ARTIFACTS_DIR: Path = field(init=False)
    MODEL_DIR: Path = field(init=False)
    REPORTS_DIR: Path = field(init=False)

    # Параметры обработки данных
    MAX_ROWS: Optional[int] = None

    # Параметры TF-IDF
    TFIDF_NGRAM_RANGE: Tuple[int, int] = (3, 5)
    TFIDF_MIN_DF: int = 1
    TFIDF_MAX_FEATURES: int = 10000

    # Параметры классификации
    KNN_K: int = 10
    MIN_CONFIDENCE: float = 0.35
    MIN_TOP_SIMILARITY: float = 0.20
    MIN_LABELED_SAMPLES: int = 5

    # Параметры BERT
    BERT_MODEL_NAME: str = "DeepPavlov/rubert-base-cased"
    BERT_TEST_SIZE_STANDARD: float = 0.20
    BERT_TEST_SIZE_ENHANCED: float = 0.10
    BERT_MAX_LENGTH_STANDARD: int = 64
    BERT_MAX_LENGTH_ENHANCED: int = 128
    BERT_BATCH_SIZE_STANDARD: int = 8
    BERT_BATCH_SIZE_ENHANCED: int = 16
    BERT_NUM_EPOCHS_STANDARD: int = 3
    BERT_NUM_EPOCHS_ENHANCED: int = 7
    BERT_LEARNING_RATE: float = 2e-5
    BERT_MIN_SAMPLES_PER_CLASS_STANDARD: int = 10
    BERT_MIN_SAMPLES_PER_CLASS_ENHANCED: int = 5

    # Регулярное выражение для валидации кодов ОКПД2
    OKPD2_PATTERN: str = r"^\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?$"

    # Поиск колонок в исходных файлах
    COLUMN_MAPPINGS: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "id": [
                "идентификатор номенклатуры",
                "id",
                "артикул",
                "item",
                "номенк",
            ],
            "name": [
                "Номенклатура",
                "номенклатура",
                "наименование",
                "наиме",
                "товар",
                "name",
            ],
            "okpd": [
                "Код ОКПД2",
                "код окпд2",
                "окпд2",
                "окпд",
                "okpd2",
                "okpd",
            ],
            "vat": [
                "ставка ндс (заказ)",
                "ставка ндс",
                "ндс",
                "vat",
            ],
            "okpd_reference": [
                "окпд2_эталон",
                "окпд эталон",
                "эталон окпд2",
                "эталон окпд",
            ],
            "vat_reference": [
                "ставка ндс_эталон",
                "ставка ндс эталон",
                "эталон ндс",
            ],
        }
    )

    # Паттерны для очистки текста
    WEIGHT_PATTERNS: List[str] = field(
        default_factory=lambda: [
            r"\b\d+\s?(г|гр|грамм|кг|мл|л)\b",
            r"\b\d+\s?/?\s?\d+\b",
        ]
    )

    # Маппинг латинских символов на кириллические
    LAT_TO_CYR_MAP: Dict[str, str] = field(
        default_factory=lambda: {
            "A": "А",
            "B": "В",
            "C": "С",
            "E": "Е",
            "H": "Н",
            "K": "К",
            "M": "М",
            "O": "О",
            "P": "Р",
            "T": "Т",
            "X": "Х",
            "Y": "У",
        }
    )

    def __post_init__(self) -> None:
        self.DATA_DIR = self.BASE_DIR / "data"
        self.TO_PROCESS_DIR = self.DATA_DIR / "to_process"
        self.TRAINING_DIR = self.DATA_DIR / "training"
        self.REFERENCE_DIR = self.DATA_DIR / "reference"
        self.RESULTS_DIR = self.DATA_DIR / "results"
        self.ARCHIVE_DIR = self.BASE_DIR / "archive"

        self.PRODUCTS_FILE = self.TO_PROCESS_DIR / "Номенклатурные единицы.xlsx"
        self.MERGED_PRODUCTS_FILE = self.TRAINING_DIR / "Номенклатурные единицы_merged.xlsx"
        self.ABBREVIATIONS_FILE = self.REFERENCE_DIR / "сокращения.xlsx"
        self.PP908_FILE = (
            self.REFERENCE_DIR
            / "Постановление Правительства РФ от 31.12.2004 N 908 Об утверждении перечней кодов видов.rtf"
        )
        self.VAT_REFERENCE_FILE = self.REFERENCE_DIR / "Товары ставка 10%.xlsx"

        self.OUTPUT_DIR = self.RESULTS_DIR
        self.BASELINE_OUTPUT_DIR = self.OUTPUT_DIR / "baseline"
        self.BERT_OUTPUT_DIR = self.OUTPUT_DIR / "bert"
        self.ARTIFACTS_DIR = self.BASE_DIR / "artifacts"
        self.MODEL_DIR = self.ARTIFACTS_DIR / "models" / "bert_okpd"
        self.REPORTS_DIR = self.ARTIFACTS_DIR / "reports"


config = Config()
