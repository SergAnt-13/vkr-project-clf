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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import numpy as np

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

            # Находим классы с <2 примерами (их нельзя стратифицировать)
            class_counts = labeled_df['okpd2_current'].value_counts()
            rare_classes = class_counts[class_counts < 2].index.tolist()

            rare_mask = labeled_df['okpd2_current'].isin(rare_classes)
            rare_df = labeled_df[rare_mask].copy()
            rest_df = labeled_df[~rare_mask].copy()

            if len(rest_df) > 0:
                train_rest, test_df = train_test_split(
                    rest_df, test_size=0.2, random_state=42,
                    stratify=rest_df['okpd2_current']
                )
                train_df = pd.concat([train_rest, rare_df], ignore_index=True)
            else:
                logger.warning("Все классы имеют менее 2 примеров, валидация невозможна.")
                return {"metrics": {"accuracy": 0, "f1_macro": 0, "f1_weighted": 0}, "validation": True}

            # Обучение на train
            classifier.fit(train_df, text_column="name_norm")

            # Предсказание на test (с уверенностями)
            test_preds = classifier.predict(test_df, text_column="name_norm")
            test_df["okpd2_pred"] = test_preds["okpd2_pred"].values
            test_df["conf"] = test_preds["conf"].values

            # Метрики
            y_true = test_df["okpd2_current"].astype(str).str.strip()
            y_pred = test_df["okpd2_pred"].astype(str).str.strip()
            y_pred_clean = y_pred.replace("", "unknown")

            from sklearn.metrics import accuracy_score, f1_score, classification_report
            accuracy = accuracy_score(y_true, y_pred_clean)
            f1_macro = f1_score(y_true, y_pred_clean, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred_clean, average='weighted', zero_division=0)
            cls_report = classification_report(y_true, y_pred_clean, output_dict=True, zero_division=0)

            logger.info("=== ЧЕСТНЫЕ МЕТРИКИ НА ОТЛОЖЕННОЙ ВЫБОРКЕ (Baseline) ===")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 (macro): {f1_macro:.4f}")
            logger.info(f"F1 (weighted): {f1_weighted:.4f}")

            # ================== ИЕРАРХИЧЕСКИЙ АНАЛИЗ ОШИБОК ==================
            def hierarchical_accuracy(y_true, y_pred, level):
                """Точность на уровне: 1 - класс (2 знака), 2 - группа (5 знаков), 3 - полный код"""
                correct = 0
                total = 0
                for t, p in zip(y_true, y_pred):
                    if t == "" or p == "" or t == "unknown" or p == "unknown":
                        continue
                    if level == 1:
                        if t[:2] == p[:2]:
                            correct += 1
                    elif level == 2:
                        if len(t) >= 5 and len(p) >= 5 and t[:5] == p[:5]:
                            correct += 1
                    elif level == 3:
                        if t == p:
                            correct += 1
                    total += 1
                return correct / total if total > 0 else 0

            acc_class = hierarchical_accuracy(y_true, y_pred, level=1)
            acc_group = hierarchical_accuracy(y_true, y_pred, level=2)
            acc_full = hierarchical_accuracy(y_true, y_pred, level=3)

            logger.info(
                f"Иерархическая точность: класс={acc_class:.4f}, группа={acc_group:.4f}, полный код={acc_full:.4f}")

            # ================== ГРАФИК УВЕРЕННОСТИ ==================
            import matplotlib.pyplot as plt
            import seaborn as sns
            test_df["is_correct"] = y_true == y_pred
            plt.figure(figsize=(10, 6))
            sns.histplot(data=test_df, x="conf", hue="is_correct", bins=30, element="step", stat="density",
                         common_norm=False)
            plt.title("Распределение уверенности (Baseline)")
            plt.xlabel("Уверенность")
            plt.ylabel("Плотность")
            conf_plot_path = runtime.OUTPUT_DIR / "confidence_distribution_baseline.png"
            plt.savefig(conf_plot_path)
            plt.close()
            logger.info(f"График уверенности сохранён: {conf_plot_path}")

            # ================== МАТРИЦА ОШИБОК (топ-15) ==================
            top15_classes = class_counts.nlargest(15).index.tolist()
            mask = y_true.isin(top15_classes) & y_pred.isin(top15_classes)
            y_true_f = y_true[mask]
            y_pred_f = y_pred[mask]
            if len(y_true_f) > 0:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true_f, y_pred_f, labels=top15_classes)
                plt.figure(figsize=(14, 12))
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=top15_classes, yticklabels=top15_classes)
                plt.title('Матрица ошибок (топ-15 классов) — Baseline')
                plt.xlabel('Предсказанный код')
                plt.ylabel('Истинный код')
                cm_plot_path = runtime.OUTPUT_DIR / 'confusion_matrix_top15_baseline.png'
                plt.savefig(cm_plot_path)
                plt.close()
                logger.info(f"Матрица ошибок сохранена: {cm_plot_path}")

            # ================== СОХРАНЕНИЕ ОТЧЁТА ==================
            report_lines = [
                "=== ВАЛИДАЦИОННЫЙ ОТЧЁТ (Baseline) ===",
                f"Дата: {pd.Timestamp.now()}",
                f"Обучающая выборка: {len(train_df)} записей",
                f"Тестовая выборка: {len(test_df)} записей",
                "",
                "Метрики на тестовой выборке:",
                f"Accuracy (Точность): {accuracy:.4f}",
                f"F1 (макро): {f1_macro:.4f}",
                f"F1 (взвешенная): {f1_weighted:.4f}",
                "",
                "Иерархическая точность:",
                f"  Уровень класса (XX): {acc_class:.4f}",
                f"  Уровень группы (XX.XX): {acc_group:.4f}",
                f"  Полный код (XX.XX.XX.XXX): {acc_full:.4f}",
                "",
                "Лучшие классы по F1 (из CSV):"
            ]

            # Топ-3 лучших/худших классов
            cls_report_df = pd.DataFrame(cls_report).transpose()
            cls_metrics = cls_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            if 'f1-score' in cls_metrics.columns:
                best = cls_metrics.nlargest(3, 'f1-score')
                worst = cls_metrics.nsmallest(3, 'f1-score')
                for idx, row in best.iterrows():
                    report_lines.append(f"  {idx}: f1={row['f1-score']:.2f}, support={int(row['support'])}")
                report_lines.append("Худшие классы по F1:")
                for idx, row in worst.iterrows():
                    report_lines.append(f"  {idx}: f1={row['f1-score']:.2f}, support={int(row['support'])}")

            # Сохраняем CSV с детальным отчётом
            cls_report_df.to_csv(runtime.OUTPUT_DIR / "classification_report_baseline.csv", encoding='utf-8-sig')
            report_path = runtime.OUTPUT_DIR / "validation_report_baseline.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            logger.info(f"Валидационный отчёт сохранён: {report_path}")

            return {"metrics": {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted},
                    "validation": True}

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
            # Честная валидация через внутренний split BERT
            fit_kwargs = self._bert_fit_kwargs(runtime, mode)
            metrics = classifier.fit(
                training_df,
                text_column="name_norm",
                label_column=label_column,
                save_path=str(runtime.MODEL_DIR) if not load_existing_model else None,
                **fit_kwargs,
            )

            # Получаем предсказания на тестовой выборке, которую BERT сохранил внутри
            classifier.set_temperature(2.0)
            y_true, y_pred_str, confidences = classifier.get_test_predictions()
            y_true_series = pd.Series(y_true)
            y_pred_series = pd.Series(y_pred_str)

            # Метрики уже есть в metrics, но пересчитаем для уверенности
            accuracy = metrics.get('accuracy', 0)
            f1_weighted = metrics.get('weighted_f1', 0)
            f1_macro = metrics.get('macro_f1', 0)

            # Иерархическая точность
            def hierarchical_accuracy(y_true, y_pred, level):
                correct, total = 0, 0
                for t, p in zip(y_true, y_pred):
                    if t in ("", "unknown") or p in ("", "unknown"):
                        continue
                    if level == 1 and t[:2] == p[:2]:
                        correct += 1
                    elif level == 2 and len(t) >= 5 and len(p) >= 5 and t[:5] == p[:5]:
                        correct += 1
                    elif level == 3 and t == p:
                        correct += 1
                    total += 1
                return correct / total if total > 0 else 0

            acc_class = hierarchical_accuracy(y_true, y_pred_str, level=1)
            acc_group = hierarchical_accuracy(y_true, y_pred_str, level=2)
            acc_full = accuracy  # полный код совпадает с accuracy

            # График уверенности
            import matplotlib.pyplot as plt
            import seaborn as sns
            test_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_str, 'conf': confidences})
            test_df['is_correct'] = test_df['y_true'] == test_df['y_pred']
            plt.figure(figsize=(10, 6))
            sns.histplot(data=test_df, x='conf', hue='is_correct', bins=30, element='step', stat='density',
                         common_norm=False)
            plt.title('Распределение уверенности (BERT)')
            plt.xlabel('Уверенность')
            plt.ylabel('Плотность')
            conf_plot_path = runtime.OUTPUT_DIR / 'confidence_distribution_bert.png'
            plt.savefig(conf_plot_path)
            plt.close()

            # Матрица ошибок для топ-15
            from sklearn.metrics import confusion_matrix
            top15 = y_true_series.value_counts().nlargest(15).index.tolist()
            mask = y_true_series.isin(top15) & y_pred_series.isin(top15)
            y_true_f = y_true_series[mask]
            y_pred_f = y_pred_series[mask]
            if len(y_true_f) > 0:
                cm = confusion_matrix(y_true_f, y_pred_f, labels=top15)
                plt.figure(figsize=(14, 12))
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=top15, yticklabels=top15)
                plt.title('Матрица ошибок (топ-15 классов) — BERT')
                plt.xlabel('Предсказанный код')
                plt.ylabel('Истинный код')
                cm_plot_path = runtime.OUTPUT_DIR / 'confusion_matrix_top15_bert.png'
                plt.savefig(cm_plot_path)
                plt.close()

            # Сохранение отчёта
            report_lines = [
                "=== ВАЛИДАЦИОННЫЙ ОТЧЁТ (BERT) ===",
                f"Дата: {pd.Timestamp.now()}",
                f"Режим: {mode}",
                f"Количество классов: {metrics.get('num_classes', '?')}",
                f"Примеров всего: {metrics.get('num_samples', '?')}",
                "",
                "Метрики на тестовой выборке:",
                f"Accuracy (Точность): {accuracy:.4f}",
                f"F1 (взвешенная): {f1_weighted:.4f}",
                f"F1 (макро): {f1_macro:.4f}",
                "",
                "Иерархическая точность:",
                f"  Уровень класса (XX): {acc_class:.4f}",
                f"  Уровень группы (XX.XX): {acc_group:.4f}",
                f"  Полный код (XX.XX.XX.XXX): {acc_full:.4f}",
            ]

            # CSV отчёт по классам
            cls_report = metrics.get('classification_report')
            if cls_report:
                cls_report_df = pd.DataFrame(cls_report).transpose()
                cls_report_df.to_csv(runtime.OUTPUT_DIR / "classification_report_bert.csv", encoding='utf-8-sig')
                cls_metrics = cls_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
                if 'f1-score' in cls_metrics.columns:
                    best = cls_metrics.nlargest(3, 'f1-score')
                    worst = cls_metrics.nsmallest(3, 'f1-score')
                    report_lines.append("")
                    report_lines.append("Лучшие классы по F1:")
                    for idx, row in best.iterrows():
                        report_lines.append(f"  {idx}: f1={row['f1-score']:.2f}, support={int(row['support'])}")
                    report_lines.append("Худшие классы по F1:")
                    for idx, row in worst.iterrows():
                        report_lines.append(f"  {idx}: f1={row['f1-score']:.2f}, support={int(row['support'])}")

            report_path = runtime.OUTPUT_DIR / "validation_report_bert.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            logger.info(f"Валидационный отчёт сохранён: {report_path}")

            return {"metrics": metrics, "validation": True}

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

        classifier.set_temperature(2.0)
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
