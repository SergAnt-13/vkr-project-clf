"""
Модуль для обработки и нормализации текста
"""
import pandas as pd
import unicodedata
import re
import logging
from typing import List, Tuple
from config import config


logger = logging.getLogger(__name__)


class TextProcessor:
    """Класс для обработки и нормализации текста"""
    
    def __init__(self, abbreviation_rules: List[Tuple[str, str]] = None, config_obj=None):
        if config_obj is None and abbreviation_rules is not None and not isinstance(abbreviation_rules, list):
            config_obj = abbreviation_rules
            abbreviation_rules = None
        
        self.config = config_obj or config
        self.abbreviation_rules = abbreviation_rules or []
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компилировать регулярные выражения для ускорения работы"""
        self.weight_patterns = [re.compile(pattern) for pattern in self.config.WEIGHT_PATTERNS]
        self.punctuation_pattern = re.compile(r"[^\w\s]")
        self.whitespace_pattern = re.compile(r"\s+")
    
    def replace_lookalikes(self, text: str) -> str:
        """Заменить похожие латинские символы на кириллические"""
        if not isinstance(text, str):
            return str(text)
        
        translator = str.maketrans(self.config.LAT_TO_CYR_MAP)
        return unicodedata.normalize("NFC", text).translate(translator)

    def apply_abbreviations(self, text: str) -> str:
        """Применить правила сокращений токен-за-токеном, чтобы избежать замен внутри слов."""
        if not self.abbreviation_rules:
            return text

        tokens = re.findall(r'\b\w+(?:\.\w+)+\b|\b\w+\b|[^\w\s]', text)
        result_tokens = []
        for token in tokens:
            token_lower = token.lower().strip('.')
            replaced = None
            for rule in self.abbreviation_rules:
                if isinstance(rule, dict):
                    pattern = str(rule.get('pattern', '')).lower().strip('.')
                    replacement = str(rule.get('replacement', ''))
                else:
                    pattern = str(rule[0]).lower().strip('.') if len(rule) > 0 else ''
                    replacement = str(rule[1]) if len(rule) > 1 else ''
                if token_lower == pattern:
                    replaced = replacement
                    break
            result_tokens.append(replaced if replaced else token)
        return ' '.join(result_tokens)

    def normalize_text_light(self, text: str) -> str:
        """Легкая нормализация для semantic pipeline: сохраняет предлоги и порядок слов."""
        if pd.isna(text) or text is None:
            return ""

        text = str(text).strip().lower()
        if not text:
            return ""

        text = self.replace_lookalikes(text)
        if self.abbreviation_rules:
            text = self.apply_abbreviations(text)
        text = self.whitespace_pattern.sub(" ", text)
        return text.strip()

    def normalize_text_light_series(self, series: pd.Series) -> pd.Series:
        logger.info(f"Легкая нормализация {len(series)} текстов")
        return series.apply(self.normalize_text_light)
    
    def clean_weights_and_measures(self, text: str) -> str:
        """Удалить упоминания весов и мер из текста"""
        result = text
        for pattern in self.weight_patterns:
            result = pattern.sub(" ", result)
        return result
    
    def clean_punctuation(self, text: str) -> str:
        """Удалить пунктуацию и нормализовать пробелы"""
        # Удалить пунктуацию, но оставить основные символы
        text = self.punctuation_pattern.sub(" ", text)
        # Нормализовать пробелы
        text = self.whitespace_pattern.sub(" ", text)
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """Полная нормализация текста"""
        if pd.isna(text) or text is None:
            return ""
        
        # Сохраняем исходный текст
        original_text = str(text).strip()
        
        # Приведение к строке и нижнему регистру
        text = original_text.lower()
        
        # Если текст пустой, возвращаем как есть
        if not text:
            return ""
        
        # Замена похожих символов
        text = self.replace_lookalikes(text)
        
        # Применение сокращений (только если есть правила)
        if self.abbreviation_rules:
            text = self.apply_abbreviations(text)
        
        # Очистка весов и мер
        text = self.clean_weights_and_measures(text)
        
        # Очистка пунктуации
        text = self.clean_punctuation(text)
        
        # Если результат слишком короткий, возвращаем исходный текст с минимальной обработкой
        if len(text) < 2 and len(original_text) >= 2:
            # Только убираем пунктуацию и нормализуем пробелы из исходного текста
            minimal_clean = original_text.lower()
            minimal_clean = self.punctuation_pattern.sub(" ", minimal_clean)
            minimal_clean = self.whitespace_pattern.sub(" ", minimal_clean).strip()
            if len(minimal_clean) >= 2:
                return minimal_clean
        
        return text
    
    def normalize_text_series(self, series: pd.Series) -> pd.Series:
        """Нормализовать серию текстов"""
        logger.info(f"Нормализация {len(series)} текстов")
        
        # Отладочная информация: примеры исходных текстов
        sample_original = series.head(3).tolist()
        logger.info(f"Примеры исходных текстов: {sample_original}")
        
        # Применяем нормализацию ко всей серии
        normalized = series.apply(self.normalize_text)
        
        # Отладочная информация: примеры нормализованных текстов
        sample_normalized = normalized.head(3).tolist()
        logger.info(f"Примеры нормализованных текстов: {sample_normalized}")
        
        # Статистика пустых текстов
        empty_count = (normalized == "").sum()
        logger.info(f"Пустых текстов после нормализации: {empty_count} из {len(normalized)}")
        
        logger.info("Нормализация текста завершена")
        return normalized
    
    def get_statistics(self, original_series: pd.Series, normalized_series: pd.Series) -> dict:
        """Получить статистику по обработке текста"""
        stats = {
            'total_texts': len(original_series),
            'empty_after_normalization': (normalized_series == "").sum(),
            'average_length_before': original_series.astype(str).str.len().mean(),
            'average_length_after': normalized_series.str.len().mean(),
            'unique_before': original_series.nunique(),
            'unique_after': normalized_series.nunique()
        }
        return stats
