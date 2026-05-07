"""
Модуль для загрузки и первичной подготовки данных.
"""

import logging
from pathlib import Path
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config


logger = logging.getLogger(__name__)


class DataLoader:
    """Загрузка Excel/CSV-файлов и приведение к общей схеме колонок."""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def read_table(self, file_path: Path) -> pd.DataFrame:
        """Прочитать таблицу из Excel/CSV/TSV."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        suffix = path.suffix.lower()
        logger.info("Читаем файл %s", path)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(path, sep=None, engine="python")

        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")

    def _find_column(self, candidates: List[str], columns: List[str]) -> Optional[str]:
        """Найти первую колонку, содержащую одну из подсказок."""
        lowered = [str(col).strip().lower() for col in columns]
        for candidate in candidates:
            candidate_lower = candidate.lower()
            for original, lowered_name in zip(columns, lowered):
                if candidate_lower in lowered_name:
                    return str(original)
        return None

    def _resolve_columns(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """Сопоставить реальные колонки таблицы со стандартными полями."""
        mappings: Dict[str, Optional[str]] = {}

        for key, hints in self.config.COLUMN_MAPPINGS.items():
            mappings[key] = self._find_column(hints, columns)

        if not mappings["name"] and columns:
            for column in columns:
                column_lower = str(column).strip().lower()
                if not any(word in column_lower for word in ["id", "код", "номер"]):
                    mappings["name"] = str(column)
                    break
            if not mappings["name"]:
                mappings["name"] = str(columns[0])

        return mappings

    def standardize_products(
        self,
        df: pd.DataFrame,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Привести DataFrame к стандартным полям проекта."""
        source = df.copy()
        source.columns = [str(col).strip() for col in source.columns]
        mappings = self._resolve_columns(list(source.columns))

        result = pd.DataFrame(index=source.index)

        if mappings["name"]:
            result["name_raw"] = source[mappings["name"]]
        if mappings["id"]:
            result["item_id"] = source[mappings["id"]]
        if mappings["okpd"]:
            result["okpd2_current"] = source[mappings["okpd"]]
        if mappings["vat"]:
            result["vat_current"] = source[mappings["vat"]]
        if mappings["okpd_reference"]:
            result["okpd2_reference"] = source[mappings["okpd_reference"]]
        if mappings["vat_reference"]:
            result["vat_reference"] = source[mappings["vat_reference"]]

        if "name_raw" not in result.columns:
            raise ValueError("Не удалось определить колонку с названием товара")

        result = self._ensure_required_columns(result)

        if max_rows is not None:
            result = result.head(max_rows).copy()

        return result

    def load_products(
        self,
        file_path: Optional[Path] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Загрузить основную номенклатуру."""
        path = Path(file_path) if file_path else self.config.PRODUCTS_FILE
        df = self.read_table(path)
        standardized = self.standardize_products(df, max_rows=max_rows or self.config.MAX_ROWS)
        logger.info("Загружено %s строк из %s", len(standardized), path)
        return standardized

    def load_training_data(
        self,
        file_path: Path,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Загрузить внешний обучающий файл и привести к общей схеме."""
        df = self.read_table(file_path)
        standardized = self.standardize_products(df, max_rows=max_rows)

        if "okpd2_reference" in standardized.columns:
            standardized["okpd2_current"] = standardized["okpd2_reference"].fillna(
                standardized["okpd2_current"]
            )

        if "vat_reference" in standardized.columns:
            standardized["vat_current"] = standardized["vat_reference"].fillna(
                standardized["vat_current"]
            )

        return standardized

    def load_merged_products(
        self,
        file_path: Optional[Path] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Загрузить merged-файл с эталонными колонками."""
        path = Path(file_path) if file_path else self.config.MERGED_PRODUCTS_FILE
        df = self.read_table(path)
        standardized = self.standardize_products(df, max_rows=max_rows or self.config.MAX_ROWS)
        logger.info("Загружено %s строк из merged-файла %s", len(standardized), path)
        return standardized

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить обязательные колонки, если их нет."""
        result = df.copy()

        if "item_id" not in result.columns:
            result.insert(0, "item_id", np.arange(1, len(result) + 1))

        if "okpd2_current" not in result.columns:
            result["okpd2_current"] = ""

        if "vat_current" not in result.columns:
            result["vat_current"] = ""

        if "okpd2_reference" not in result.columns:
            result["okpd2_reference"] = ""

        if "vat_reference" not in result.columns:
            result["vat_reference"] = ""

        return result

    def load_abbreviations(self) -> List[Tuple[str, str]]:
        """Загрузить правила сокращений из Excel-файла."""
        path = self.config.ABBREVIATIONS_FILE
        if not path.exists():
            logger.warning("Файл сокращений не найден: %s", path)
            return []

        try:
            df = pd.read_excel(path)
        except Exception as exc:
            logger.error("Не удалось прочитать файл сокращений: %s", exc)
            return []

        df.columns = [str(col).strip().lower() for col in df.columns]

        if len(df.columns) >= 2:
            pattern_col, replacement_col = df.columns[:2]
            rules = [
                (self._normalize_abbreviation_pattern(pattern), str(replacement).strip())
                for pattern, replacement in df[[pattern_col, replacement_col]].dropna().values
            ]
            rules = [(pattern, replacement) for pattern, replacement in rules if pattern and replacement]
            logger.info("Загружено %s правил сокращений", len(rules))
            return rules

        only_column = df.columns[0]
        parsed_rules: List[Tuple[str, str]] = []

        for raw_value in df[only_column].dropna():
            parts = [part.strip() for part in str(raw_value).split(";") if part.strip()]
            if len(parts) < 2:
                continue
            pattern = self._normalize_abbreviation_pattern(parts[0])
            replacement = parts[1]
            if pattern and replacement:
                parsed_rules.append((pattern, replacement))

        logger.info("Загружено %s правил сокращений из одноколоночного файла", len(parsed_rules))
        return parsed_rules

    def _normalize_abbreviation_pattern(self, raw_pattern: object) -> str:
        """Подготовить шаблон сокращения к использованию в regex."""
        pattern = str(raw_pattern).strip()
        if pattern.lower().startswith("pattern:"):
            pattern = pattern.split(":", 1)[1].strip()

            return pattern

        alternatives = [part.strip() for part in re.split(r"\s*/\s*", pattern) if part.strip()]
        if "/" in pattern and alternatives and all(len(part) <= 1 for part in alternatives):
            alternatives = [pattern]
        escaped = [re.escape(part) for part in alternatives]
        combined = "|".join(escaped)

        if alternatives and all(re.fullmatch(r"[\w.\-]+", part, flags=re.UNICODE) for part in alternatives):
            return rf"\b(?:{combined})\b"

        return rf"(?:{combined})"

    def load_pp908_text(self, file_path: Optional[Path] = None) -> str:
        """Загрузить текст постановления ПП-908."""
        path = Path(file_path) if file_path else self.config.PP908_FILE
        if not path.exists():
            logger.warning("Файл ПП-908 не найден: %s", path)
            return ""

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(self.config.OKPD2_PATTERN, text):
                logger.info("Загружен текст ПП-908 длиной %s символов", len(text))
                return text

            converted = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            text = converted.stdout
            logger.info("RTF преобразован в текст через textutil, длина %s символов", len(text))
            return text
        except Exception as exc:
            logger.error("Ошибка загрузки ПП-908: %s", exc)
            return ""

    def describe_source(self, file_path: Path) -> Dict[str, object]:
        """Вернуть компактное описание файла для CLI и документации."""
        df = self.read_table(file_path)
        standardized = self.standardize_products(df)

        valid_current = standardized["okpd2_current"].astype(str).str.match(self.config.OKPD2_PATTERN).fillna(False)
        valid_reference = standardized["okpd2_reference"].astype(str).str.match(self.config.OKPD2_PATTERN).fillna(False)

        return {
            "path": str(file_path),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "valid_okpd_current": int(valid_current.sum()),
            "valid_okpd_reference": int(valid_reference.sum()),
            "unique_current_codes": int(
                standardized.loc[valid_current, "okpd2_current"].astype(str).nunique()
            ),
            "unique_reference_codes": int(
                standardized.loc[valid_reference, "okpd2_reference"].astype(str).nunique()
            ),
        }
