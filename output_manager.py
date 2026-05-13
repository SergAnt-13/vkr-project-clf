"""
Модуль для управления выводом результатов
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
from config import config


logger = logging.getLogger(__name__)


class OutputManager:
    """Класс для управления сохранением и отображением результатов"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.output_dir = self.config.OUTPUT_DIR
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Убедиться, что директория для вывода существует"""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Директория для вывода: {self.output_dir}")
    
    def create_analysis_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Создать таблицы для анализа результатов"""
        base_columns = [
            "item_id", "name_raw", "okpd2_current", "okpd2_pred", 
            "conf", "okpd2_final", "vat_current", "vat_pred"
        ]
        
        # Убедимся, что все колонки присутствуют
        available_columns = [col for col in base_columns if col in df.columns]
        
        # Таблица 1: Неверные коды ОКПД2
        # Записи с валидными текущими кодами, для которых модель предлагает иной код
        table1_mask = (
            df["okpd2_current"].apply(self._is_valid_code) & 
            (df["okpd2_pred"].astype(str).str.strip() != "") &
            (df["okpd2_pred"] != df["okpd2_current"])
        )
        table1 = df[table1_mask][available_columns].copy()
        
        # Таблица 2: Неверные ставки НДС
        # Записи с заполненными текущими НДС, но отличающимися предсказанными
        table2_mask = (
            (df["vat_current"].astype(str).str.strip() != "") & 
            (df["vat_current"] != df["vat_pred"])
        )
        table2 = df[table2_mask][available_columns].copy()
        
        # Таблица 3: Остальные записи
        # Исключаем записи, попавшие в первые две таблицы
        used_indices = set(table1.index) | set(table2.index)
        table3_mask = ~df.index.isin(used_indices)
        table3 = df[table3_mask][available_columns].copy()
        
        tables = {
            "wrong_okpd2": table1,
            "wrong_vat": table2,
            "clean": table3
        }
        
        logger.info(f"Создано таблиц: неверные ОКПД2={len(table1)}, неверные НДС={len(table2)}, остальные={len(table3)}")
        
        return tables
    
    def _is_valid_code(self, code) -> bool:
        """Проверить валидность кода ОКПД2"""
        import re
        if not isinstance(code, str):
            return False
        pattern = re.compile(self.config.OKPD2_PATTERN)
        return bool(pattern.match(code.strip()))
    
    def save_tables(self, tables: Dict[str, pd.DataFrame], df_full: pd.DataFrame) -> Dict[str, Path]:
        """Сохранить таблицы в CSV файлы"""
        saved_files = {}
        
        # Сохранение аналитических таблиц
        table_configs = {
            "wrong_okpd2": ("table1_wrong_okpd2.csv", "Таблица 1 (неверные ОКПД2)"),
            "wrong_vat": ("table2_wrong_vat.csv", "Таблица 2 (неверные НДС)"),
            "clean": ("table3_clean.csv", "Таблица 3 (остальные)")
        }
        
        for table_name, (filename, description) in table_configs.items():
            if table_name in tables:
                file_path = self.output_dir / filename
                tables[table_name].to_csv(file_path, index=False, encoding='utf-8-sig')
                saved_files[table_name] = file_path
                logger.info(f"Сохранена {description}: {file_path} ({len(tables[table_name])} записей)")
        
        # Сохранение полной таблицы
        full_file_path = self.output_dir / "full_output.csv"
        df_full.to_csv(full_file_path, index=False, encoding='utf-8-sig')
        saved_files["full"] = full_file_path
        logger.info(f"Сохранена полная таблица: {full_file_path} ({len(df_full)} записей)")
        
        return saved_files
    
    def display_preview(self, tables: Dict[str, pd.DataFrame], preview_rows: int = 20):
        """Отобразить превью таблиц"""
        try:
            # Попытаться использовать caas_jupyter_tools если доступно
            from caas_jupyter_tools import display_dataframe_to_user
            
            table_descriptions = {
                "wrong_okpd2": "Таблица 1 (неверные ОКПД2): превью",
                "wrong_vat": "Таблица 2 (неверные НДС): превью", 
                "clean": "Таблица 3 (остальные): превью"
            }
            
            for table_name, description in table_descriptions.items():
                if table_name in tables and len(tables[table_name]) > 0:
                    display_dataframe_to_user(description, tables[table_name].head(preview_rows))
                    
        except ImportError:
            # Если caas_jupyter_tools недоступно, выводим обычным способом
            logger.info("caas_jupyter_tools недоступно, используем стандартный вывод")
            
            for table_name, table in tables.items():
                if len(table) > 0:
                    print(f"\n=== {table_name.upper()} ===")
                    print(f"Всего записей: {len(table)}")
                    print(table.head(preview_rows).to_string())
    
    def generate_summary_report(self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame], 
                              statistics: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Сгенерировать сводный отчет"""
        metadata = metadata or {}
        report_lines = [
            "=== СВОДНЫЙ ОТЧЕТ ===",
            f"Обработано записей: {len(df)}",
            "",
            "=== ИСТОЧНИКИ ===",
            f"Файл для предсказаний: {metadata.get('prediction_source', 'не указан')}",
            f"Файл для обучения: {metadata.get('training_source', 'не указан')}",
            "",
            "=== СТАТИСТИКА ПО ТАБЛИЦАМ ===",
            f"Неверные ОКПД2: {len(tables.get('wrong_okpd2', []))}",
            f"Неверные НДС: {len(tables.get('wrong_vat', []))}",
            f"Остальные: {len(tables.get('clean', []))}",
            ""
        ]
        
        # Добавляем статистику классификации если есть
        if 'classification' in statistics:
            cls_stats = statistics['classification']
            report_lines.extend([
                "=== СТАТИСТИКА КЛАССИФИКАЦИИ ===",
                f"Всего предсказаний: {cls_stats.get('total_predictions', 0)}",
                f"Валидных предсказаний: {cls_stats.get('valid_predictions', 0)}",
                f"Процент покрытия: {cls_stats.get('prediction_rate', 0):.2%}",
                f"Средняя уверенность: {cls_stats.get('average_confidence', 0):.3f}",
                ""
            ])
        
        # Добавляем статистику НДС если есть
        if 'vat' in statistics:
            vat_stats = statistics['vat']
            report_lines.extend([
                "=== СТАТИСТИКА НДС ===",
                f"НДС 10%: {vat_stats.get('vat10_count', 0)} ({vat_stats.get('vat10_rate', 0):.2%})",
                f"НДС 20%: {vat_stats.get('vat20_count', 0)} ({vat_stats.get('vat20_rate', 0):.2%})",
                f"Покрытие: {vat_stats.get('coverage_rate', 0):.2%}",
                ""
            ])
        # Метрики качества классификации, если они переданы
        if 'metrics' in statistics:
            metrics = statistics['metrics']
            if metrics:
                report_lines.extend([
                    "=== МЕТРИКИ КАЧЕСТВА КЛАССИФИКАЦИИ ===",
                    f"Accuracy (Точность): {metrics.get('accuracy', 0):.4f}",
                    f"F1 (макро): {metrics.get('f1_macro', 0):.4f}",
                    f"F1 (взвешенная): {metrics.get('f1_weighted', 0):.4f}",
                    ""
                ])
        report = "\n".join(report_lines)
        
        # Сохранить отчет в файл
        report_path = self.output_dir / "summary_report.txt"
        report_path.write_text(report, encoding='utf-8')
        logger.info(f"Сохранен сводный отчет: {report_path}")
        
        return report
