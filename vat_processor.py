"""
Модуль для обработки НДС на основе кодов ОКПД2 и постановления ПП-908
"""
import re
import logging
from typing import Set, List
from config import config


logger = logging.getLogger(__name__)


class VATProcessor:
    """Класс для определения ставок НДС по кодам ОКПД2"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.vat10_prefixes: Set[str] = set()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компилировать регулярные выражения"""
        self.okpd_pattern = re.compile(self.config.OKPD2_PATTERN)
        self.code_extraction_pattern = re.compile(r"\b\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?\b")
    
    def is_valid_code(self, code: str) -> bool:
        """Проверить валидность кода ОКПД2"""
        if not isinstance(code, str):
            return False
        return bool(self.okpd_pattern.match(code.strip()))
    
    def extract_codes_from_text(self, text: str) -> List[str]:
        """Извлечь все коды ОКПД2 из текста"""
        if not text:
            return []
        
        # Расширенные паттерны для поиска кодов в RTF
        patterns = [
            r"\b\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?\b",  # Основной паттерн
            r"(?:код|группа|подгруппа|класс)[\s\w]*?(\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?)",  # С контекстом
            r"(\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?)\s*(?:-|–|—)\s*[\w\s]+(?:льготн|10%|десят)",  # С упоминанием льгот
        ]
        
        all_codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if isinstance(matches[0], tuple) if matches else False:
                # Если паттерн возвращает кортежи, берем первую группу
                matches = [match[0] if isinstance(match, tuple) else match for match in matches]
            all_codes.extend(matches)
        
        # Дополнительный поиск в таблицах RTF
        table_pattern = r"\\row.*?(\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?).*?(?:льготн|10%|десят).*?\\row"
        table_matches = re.findall(table_pattern, text, re.IGNORECASE | re.DOTALL)
        all_codes.extend(table_matches)
        
        unique_codes = list(set(all_codes))
        logger.info(f"Извлечено {len(unique_codes)} уникальных кодов из текста")
        
        # Отладочная информация
        if len(unique_codes) > 0:
            logger.info(f"Примеры найденных кодов: {unique_codes[:5]}")
        else:
            # Показать фрагмент текста для диагностики
            sample_text = text[:500] + "..." if len(text) > 500 else text
            logger.warning(f"Коды не найдены. Образец текста: {sample_text}")
        
        return unique_codes
    
    def generate_prefixes(self, code: str) -> List[str]:
        """Сгенерировать все возможные префиксы кода"""
        if not self.is_valid_code(code):
            return []
        
        parts = code.strip().split(".")
        prefixes = []
        
        # Генерируем префиксы от самого короткого до самого длинного
        for i in range(2, len(parts) + 1):
            prefix = ".".join(parts[:i])
            prefixes.append(prefix)
        
        return prefixes
    
    def build_vat10_prefixes(self, pp908_text: str):
        """Построить множество префиксов для товаров с НДС 10%"""
        logger.info("Построение префиксов для НДС 10% из текста ПП-908")
        
        codes = self.extract_codes_from_text(pp908_text)
        
        vat10_prefixes = set()
        for code in codes:
            prefixes = self.generate_prefixes(code)
            vat10_prefixes.update(prefixes)
        
        self.vat10_prefixes = vat10_prefixes
        logger.info(f"Построено {len(self.vat10_prefixes)} префиксов для НДС 10%")
    
    def determine_vat_rate(self, okpd_code: str) -> str:
        """Определить ставку НДС для кода ОКПД2"""
        if not self.is_valid_code(okpd_code):
            return ""
        
        code = okpd_code.strip()
        parts = code.split(".")
        
        # Проверяем префиксы от самого длинного к самому короткому
        for i in range(len(parts), 1, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.vat10_prefixes:
                return "НДС10"
        
        # Если не найден в префиксах НДС 10%, то НДС 20%
        return "НДС20"
    
    def process_vat_predictions(self, df, okpd_column: str = "okpd2_final") -> List[str]:
        """Обработать предсказания НДС для DataFrame"""
        logger.info(f"Обработка предсказаний НДС для {len(df)} записей")
        
        vat_predictions = []
        for okpd_code in df[okpd_column]:
            vat_rate = self.determine_vat_rate(okpd_code)
            vat_predictions.append(vat_rate)
        
        valid_predictions = sum(1 for vat in vat_predictions if vat != "")
        logger.info(f"Получено {valid_predictions} валидных предсказаний НДС")
        
        return vat_predictions
    
    def get_vat_statistics(self, vat_predictions: List[str]) -> dict:
        """Получить статистику по предсказаниям НДС"""
        total = len(vat_predictions)
        vat10_count = sum(1 for vat in vat_predictions if vat == "НДС10")
        vat20_count = sum(1 for vat in vat_predictions if vat == "НДС20")
        empty_count = sum(1 for vat in vat_predictions if vat == "")
        
        stats = {
            'total_predictions': total,
            'vat10_count': vat10_count,
            'vat20_count': vat20_count,
            'empty_count': empty_count,
            'vat10_rate': vat10_count / total if total > 0 else 0,
            'vat20_rate': vat20_count / total if total > 0 else 0,
            'coverage_rate': (vat10_count + vat20_count) / total if total > 0 else 0
        }
        
        return stats
    
    def validate_vat_predictions(self, df, current_vat_column: str = "vat_current", 
                                predicted_vat_column: str = "vat_pred") -> dict:
        """Валидировать предсказания НДС против текущих значений"""
        if current_vat_column not in df.columns or predicted_vat_column not in df.columns:
            return {}
        
        # Фильтруем только те записи, где есть текущие значения НДС
        valid_current = df[df[current_vat_column].astype(str).str.strip() != ""]
        
        if len(valid_current) == 0:
            return {'message': 'Нет записей с текущими значениями НДС для валидации'}
        
        matches = (valid_current[current_vat_column] == valid_current[predicted_vat_column]).sum()
        total = len(valid_current)
        
        validation_stats = {
            'total_validated': total,
            'matches': matches,
            'mismatches': total - matches,
            'accuracy': matches / total if total > 0 else 0
        }
        
        return validation_stats
