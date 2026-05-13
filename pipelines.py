"""
Высокоуровневые пайплайны проекта.
"""

from dataclasses import replace
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from bert_okpd_classifier import BERTOKPDClassifier
from config import config
from data_loader import DataLoader
from okpd_classifier import OKPDClassifier
from output_manager import OutputManager
from text_processor import TextProcessor
from vat_processor import VATProcessor


logger = logging.getLogger(__name__)


def _valid_okpd_mask(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.strip().str.match(pattern).fillna(False)


class BaselinePipeline:
    """Базовый пайплайн TF-IDF + kNN."""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def _runtime_config(self, output_dir: Optional[Path] = None):
        runtime = replace(self.config)
        runtime.OUTPUT_DIR = Path(output_dir) if output_dir else runtime.BASELINE_OUTPUT_DIR
        return runtime

    def run(
        self,
        prediction_file: Optional[Path] = None,
        training_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        max_rows: Optional[int] = None,
        training_max_rows: Optional[int] = None,
        validate: bool = False,
    ) -> Dict[str, object]:
        runtime = self._runtime_config(output_dir)
        loader = DataLoader(runtime)
        output_manager = OutputManager(runtime)
        vat_processor = VATProcessor(runtime)
        classifier = OKPDClassifier(runtime)

        prediction_path = Path(prediction_file) if prediction_file else runtime.PRODUCTS_FILE
        prediction_df = loader.load_products(prediction_path, max_rows=max_rows)

        if training_file:
            training_path = Path(training_file)
            training_df = loader.load_merged_products(training_path, max_rows=training_max_rows)
        else:
            training_path = prediction_path
            training_df = loader.load_products(training_path, max_rows=training_max_rows)

        abbreviation_rules = loader.load_abbreviations()
        text_processor = TextProcessor(abbreviation_rules=abbreviation_rules, config_obj=runtime)

        prediction_df["name_norm"] = text_processor.normalize_text_series(prediction_df["name_raw"])
        training_df["name_norm"] = text_processor.normalize_text_series(training_df["name_raw"])

        if 'Код ОКПД2' in training_df.columns:
            training_df['okpd2_current'] = training_df['Код ОКПД2'].astype(str).str.strip()
        elif 'okpd2_current' not in training_df.columns or training_df['okpd2_current'].isna().all():
            # пробуем другие возможные названия
            for col in ['Код ОКПД2', 'код окпд2', 'ОКПД2', 'okpd2']:
                if col in training_df.columns:
                    training_df['okpd2_current'] = training_df[col].astype(str).str.strip()
                    break

        labeled_df, _ = classifier.split_labeled_unlabeled(training_df)

        if validate:
            # === ЧЕСТНАЯ ВАЛИДАЦИЯ: train/test split с учётом редких классов ===
            from sklearn.model_selection import train_test_split

            # Находим классы, у которых меньше 2 примеров (их нельзя стратифицировать)
            class_counts = labeled_df['okpd2_current'].value_counts()
            rare_classes = class_counts[class_counts < 2].index.tolist()

            # Отделяем редкие классы (они пойдут только в обучение)
            rare_mask = labeled_df['okpd2_current'].isin(rare_classes)
            rare_df = labeled_df[rare_mask].copy()
            rest_df = labeled_df[~rare_mask].copy()

            # Разбиваем остальные данные со стратификацией
            if len(rest_df) > 0:
                train_rest, test_df = train_test_split(
                    rest_df, test_size=0.2, random_state=42,
                    stratify=rest_df['okpd2_current']
                )
                train_df = pd.concat([train_rest, rare_df], ignore_index=True)
            else:
                logger.warning("Все классы имеют менее 2 примеров, валидация невозможна.")
                return {"metrics": {"accuracy": 0, "f1_macro": 0, "f1_weighted": 0}, "validation": True}

            classifier.fit(train_df, text_column="name_norm")
            eval_metrics = classifier.evaluate(test_df, text_column="name_norm")

            # Вывод в консоль
            logger.info("=== ЧЕСТНЫЕ МЕТРИКИ НА ОТЛОЖЕННОЙ ВЫБОРКЕ (Baseline) ===")
            logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}")
            logger.info(f"F1 (macro): {eval_metrics['f1_macro']:.4f}")
            logger.info(f"F1 (weighted): {eval_metrics['f1_weighted']:.4f}")

            # Сохранение отчёта в файл
            report_lines = [
                "=== ВАЛИДАЦИОННЫЙ ОТЧЁТ (Baseline) ===",
                f"Дата: {pd.Timestamp.now()}",
                f"Обучающая выборка: {len(train_df)} записей",
                f"Тестовая выборка: {len(test_df)} записей",
                "",
                "Метрики на тестовой выборке:",
                f"Accuracy (Точность): {eval_metrics['accuracy']:.4f}",
                f"F1 (макро): {eval_metrics['f1_macro']:.4f}",
                f"F1 (взвешенная): {eval_metrics['f1_weighted']:.4f}",
            ]
            report_path = runtime.OUTPUT_DIR / "validation_report_baseline.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            logger.info(f"Валидационный отчёт сохранён: {report_path}")

            return {"metrics": eval_metrics, "validation": True, "report_path": report_path}

        classifier.fit(labeled_df, text_column="name_norm")
        predictions_df = classifier.predict(prediction_df, text_column="name_norm")

        result_df = prediction_df.merge(predictions_df, on="item_id", how="left")
        result_df["okpd2_pred"] = result_df["okpd2_pred"].fillna("")
        result_df["conf"] = result_df["conf"].fillna(0.0)

        result_df["okpd2_final"] = result_df.apply(self._choose_final_okpd, axis=1)

        pp908_text = loader.load_pp908_text()
        if pp908_text:
            vat_processor.build_vat10_prefixes(pp908_text)

        result_df["vat_pred"] = vat_processor.process_vat_predictions(result_df, "okpd2_final")

        eval_metrics = {}
        if len(labeled_df) > 0:
            eval_metrics = classifier.evaluate(labeled_df, text_column="name_norm")

        tables = output_manager.create_analysis_tables(result_df)
        classification_stats = classifier.get_prediction_statistics(predictions_df)
        vat_stats = vat_processor.get_vat_statistics(result_df["vat_pred"].tolist())

        stats_for_report = {
            "classification": classification_stats,
            "vat": vat_stats,
            "metrics": eval_metrics
        }

        report = output_manager.generate_summary_report(
            result_df,
            tables,
            stats_for_report,
            metadata={
                "prediction_source": str(prediction_path),
                "training_source": str(training_path),
            },
        )
        saved_files = output_manager.save_tables(tables, result_df)
        saved_files["report"] = runtime.OUTPUT_DIR / "summary_report.txt"

        if eval_metrics:
            logger.info("=== ИТОГОВЫЕ МЕТРИКИ (baseline) ===")
            logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}")
            logger.info(f"F1 (macro): {eval_metrics['f1_macro']:.4f}")
            logger.info(f"F1 (weighted): {eval_metrics['f1_weighted']:.4f}")

        return {
            "data": result_df,
            "tables": tables,
            "report": report,
            "saved_files": saved_files,
            "prediction_source": prediction_path,
            "training_source": training_path,
        }

    def _choose_final_okpd(self, row: pd.Series) -> str:
        current = str(row.get("okpd2_current", "")).strip()
        if current and _valid_okpd_mask(pd.Series([current]), self.config.OKPD2_PATTERN).iloc[0]:
            return current

        predicted = str(row.get("okpd2_pred", "")).strip()
        if predicted and _valid_okpd_mask(pd.Series([predicted]), self.config.OKPD2_PATTERN).iloc[0]:
            return predicted

        return ""


class BERTPipeline:
    """Пайплайн для обучения и применения BERT-классификатора."""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def _runtime_config(
        self,
        output_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
    ):
        runtime = replace(self.config)
        runtime.OUTPUT_DIR = Path(output_dir) if output_dir else runtime.BERT_OUTPUT_DIR
        runtime.MODEL_DIR = Path(model_dir) if model_dir else runtime.MODEL_DIR
        runtime.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        runtime.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return runtime

    def run(
        self,
        mode: str = "enhanced",
        prediction_file: Optional[Path] = None,
        training_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        load_existing_model: bool = False,
        max_rows: Optional[int] = None,
        training_max_rows: Optional[int] = None,
        validate: bool = False,
    ) -> Dict[str, object]:
        runtime = self._runtime_config(output_dir, model_dir)
        loader = DataLoader(runtime)
        vat_processor = VATProcessor(runtime)
        classifier = BERTOKPDClassifier(runtime, model_name=runtime.BERT_MODEL_NAME)

        prediction_path = self._resolve_prediction_source(runtime, mode, prediction_file)
        prediction_raw = loader.read_table(prediction_path)
        prediction_df = loader.standardize_products(prediction_raw, max_rows=max_rows)

        training_df, training_path, label_column = self._resolve_training_source(
            loader=loader,
            mode=mode,
            training_file=training_file,
            prediction_path=prediction_path,
            training_max_rows=training_max_rows,
        )

        abbreviation_rules = loader.load_abbreviations()
        text_processor = TextProcessor(abbreviation_rules=abbreviation_rules, config_obj=runtime)

        prediction_df["name_norm"] = text_processor.normalize_text_series(prediction_df["name_raw"])
        training_df["name_norm"] = text_processor.normalize_text_series(training_df["name_raw"])
        training_df = training_df[training_df["name_norm"].astype(str).str.len() > 0].copy()

        if validate:
            # ЧЕСТНАЯ ВАЛИДАЦИЯ – только обучение и метрики на внутреннем тесте
            fit_kwargs = self._bert_fit_kwargs(runtime, mode)
            metrics = classifier.fit(
                training_df,
                text_column="name_norm",
                label_column=label_column,
                save_path=str(runtime.MODEL_DIR) if not load_existing_model else None,
                **fit_kwargs,
            )
            # Вывод в консоль
            logger.info("=== ЧЕСТНЫЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ (BERT) ===")
            logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"F1 (weighted): {metrics.get('weighted_f1', 0):.4f}")
            logger.info(f"F1 (macro): {metrics.get('macro_f1', 0):.4f}")

            # Сохранение отчёта в файл
            report_lines = [
                "=== ВАЛИДАЦИОННЫЙ ОТЧЁТ (BERT) ===",
                f"Дата: {pd.Timestamp.now()}",
                f"Режим: {mode}",
                f"Количество классов: {metrics.get('num_classes', '?')}",
                f"Примеров всего: {metrics.get('num_samples', '?')}",
                "",
                "Метрики на тестовой выборке:",
                f"Accuracy (Точность): {metrics.get('accuracy', 0):.4f}",
                f"F1 (взвешенная): {metrics.get('weighted_f1', 0):.4f}",
                f"F1 (макро): {metrics.get('macro_f1', 0):.4f}",
            ]
            report_path = runtime.OUTPUT_DIR / "validation_report_bert.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            logger.info(f"Валидационный отчёт сохранён: {report_path}")

            return {"metrics": metrics, "validation": True, "report_path": report_path}

        if load_existing_model:
            classifier.load_model(runtime.MODEL_DIR)
        else:
            fit_kwargs = self._bert_fit_kwargs(runtime, mode)
            bert_metrics = classifier.fit(
                training_df,
                text_column="name_norm",
                label_column=label_column,
                save_path=str(runtime.MODEL_DIR),
                **fit_kwargs,
            )

        predictions_df = classifier.predict_dataframe(prediction_df, text_column="name_norm")
        predictions_df["okpd2_final"] = predictions_df.apply(self._choose_final_okpd, axis=1)

        pp908_text = loader.load_pp908_text()
        if pp908_text:
            vat_processor.build_vat10_prefixes(pp908_text)

        predictions_df["vat_pred"] = vat_processor.process_vat_predictions(predictions_df, "okpd2_final")
        predictions_df["vat_final"] = predictions_df.apply(self._choose_final_vat, axis=1)

        result_output = prediction_raw.iloc[: len(predictions_df)].copy()
        result_output["Код ОКПД2 (предсказанный)"] = predictions_df["okpd2_pred"].values
        result_output["Уверенность предсказания"] = predictions_df["conf"].values
        result_output["Код ОКПД2 (финальный)"] = predictions_df["okpd2_final"].values
        result_output["Ставка НДС (предсказанная)"] = predictions_df["vat_pred"].values
        result_output["Ставка НДС (финальная)"] = predictions_df["vat_final"].values

        output_file = runtime.OUTPUT_DIR / self._output_filename(mode, prediction_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_output.to_excel(output_file, index=False)

        report_path = runtime.OUTPUT_DIR / self._report_filename(mode)
        report = self._build_report(
            predictions_df=predictions_df,
            mode=mode,
            prediction_path=prediction_path,
            training_path=training_path,
            label_column=label_column,
            output_file=output_file,
            metrics=bert_metrics if not load_existing_model else None,
        )
        report_path.write_text(report, encoding="utf-8")

        return {
            "data": predictions_df,
            "output_file": output_file,
            "report_file": report_path,
            "report": report,
            "prediction_source": prediction_path,
            "training_source": training_path,
            "model_dir": runtime.MODEL_DIR,
        }

    def _resolve_prediction_source(
        self,
        runtime,
        mode: str,
        prediction_file: Optional[Path],
    ) -> Path:
        if prediction_file:
            return Path(prediction_file)

        if mode == "enhanced" and runtime.MERGED_PRODUCTS_FILE.exists():
            return runtime.MERGED_PRODUCTS_FILE

        return runtime.PRODUCTS_FILE

    def _resolve_training_source(
        self,
        loader: DataLoader,
        mode: str,
        training_file: Optional[Path],
        prediction_path: Path,
        training_max_rows: Optional[int],
    ):
        if training_file:
            training_path = Path(training_file)
            training_df = loader.load_merged_products(training_path, max_rows=training_max_rows)
        elif mode == "enhanced" and self.config.MERGED_PRODUCTS_FILE.exists():
            training_path = self.config.MERGED_PRODUCTS_FILE
            training_df = loader.load_merged_products(training_path, max_rows=training_max_rows)
        else:
            training_path = prediction_path
            training_df = loader.load_products(training_path, max_rows=training_max_rows)

        training_df = training_df.copy()

        # Пробуем взять эталонные коды (okpd2_reference)
        valid_reference = _valid_okpd_mask(training_df["okpd2_reference"], self.config.OKPD2_PATTERN)
        if valid_reference.any():
            training_df["training_label"] = training_df["okpd2_reference"].astype(str).str.strip()
        else:
            # Если эталонных кодов нет, пробуем взять текущие коды (okpd2_current)
            valid_current = _valid_okpd_mask(training_df["okpd2_current"], self.config.OKPD2_PATTERN)
            if valid_current.any():
                training_df["training_label"] = training_df["okpd2_current"].astype(str).str.strip()
            else:
                # Если и их нет — заполняем пустыми значениями
                training_df["training_label"] = ""

        return training_df, training_path, "training_label"

    def _bert_fit_kwargs(self, runtime, mode: str) -> Dict[str, object]:
        if mode == "enhanced":
            return {
                "test_size": runtime.BERT_TEST_SIZE_ENHANCED,
                "max_length": runtime.BERT_MAX_LENGTH_ENHANCED,
                "batch_size": runtime.BERT_BATCH_SIZE_ENHANCED,
                "num_epochs": runtime.BERT_NUM_EPOCHS_ENHANCED,
                "learning_rate": runtime.BERT_LEARNING_RATE,
                "min_samples_per_class": runtime.BERT_MIN_SAMPLES_PER_CLASS_ENHANCED,
            }

        return {
            "test_size": runtime.BERT_TEST_SIZE_STANDARD,
            "max_length": runtime.BERT_MAX_LENGTH_STANDARD,
            "batch_size": runtime.BERT_BATCH_SIZE_STANDARD,
            "num_epochs": runtime.BERT_NUM_EPOCHS_STANDARD,
            "learning_rate": runtime.BERT_LEARNING_RATE,
            "min_samples_per_class": runtime.BERT_MIN_SAMPLES_PER_CLASS_STANDARD,
        }

    def _choose_final_okpd(self, row: pd.Series) -> str:
        for field in ("okpd2_reference", "okpd2_current", "okpd2_pred"):
            value = str(row.get(field, "")).strip()
            if value and _valid_okpd_mask(pd.Series([value]), self.config.OKPD2_PATTERN).iloc[0]:
                return value
        return ""

    def _choose_final_vat(self, row: pd.Series) -> str:
        for field in ("vat_reference", "vat_current", "vat_pred"):
            value = str(row.get(field, "")).strip()
            if value:
                return value
        return ""

    def _output_filename(self, mode: str, prediction_path: Path) -> str:
        suffix = "enhanced" if mode == "enhanced" else "standard"
        return f"{prediction_path.stem}_bert_{suffix}.xlsx"

    def _report_filename(self, mode: str) -> str:
        suffix = "enhanced" if mode == "enhanced" else "standard"
        return f"bert_{suffix}_report.txt"

    def _build_report(
        self,
        predictions_df: pd.DataFrame,
        mode: str,
        prediction_path: Path,
        training_path: Path,
        label_column: str,
        output_file: Path,
        metrics: Optional[Dict] = None,
    ) -> str:
        total_records = len(predictions_df)
        predicted_records = int(
            predictions_df["okpd2_pred"].astype(str).str.strip().ne("").sum()
        )
        labeled_training = int(
            predictions_df.get("okpd2_reference", pd.Series([], dtype=str))
            .astype(str)
            .str.match(self.config.OKPD2_PATTERN)
            .fillna(False)
            .sum()
        )
        confidence_stats = predictions_df["conf"].describe()
        vat_distribution = predictions_df["vat_final"].value_counts(dropna=False)

        lines = [
            f"Режим BERT: {mode}",
            f"Файл для предсказаний: {prediction_path}",
            f"Файл для обучения: {training_path}",
            f"Колонка с метками обучения: {label_column}",
            f"Выходной файл: {output_file}",
            "",
            "Статистика:",
            f"- всего записей: {total_records}",
            f"- записей с предсказанным кодом: {predicted_records}",
            f"- записей с эталонными метками: {labeled_training}",
            f"- средняя уверенность: {confidence_stats.get('mean', 0.0):.4f}",
            f"- медианная уверенность: {confidence_stats.get('50%', 0.0):.4f}",
            f"- максимальная уверенность: {confidence_stats.get('max', 0.0):.4f}",
            "",
            "Распределение НДС:",
        ]

        for vat_value, count in vat_distribution.items():
            lines.append(f"- {vat_value}: {count}")

        if metrics:
            lines.append("")
            lines.append("Метрики качества классификации:")
            lines.append(f"- Accuracy (Точность): {metrics.get('accuracy', 0.0):.4f}")
            lines.append(f"- F1 (взвешенная): {metrics.get('weighted_f1', 0.0):.4f}")
            lines.append(f"- F1 (макро): {metrics.get('macro_f1', 0.0):.4f}")
            # classification_report при желании можно не выводить, иначе очень много строк
        return "\n".join(lines)
