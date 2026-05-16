"""Baseline TF-IDF + kNN pipeline."""

from dataclasses import replace
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

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

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        runtime.OUTPUT_DIR = runtime.OUTPUT_DIR / timestamp
        runtime.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_xlsx = runtime.OUTPUT_DIR / f"{prediction_path.stem}_baseline_{ts}.xlsx"
        result_df.to_excel(output_xlsx, index=False)
        saved_files["baseline_excel"] = output_xlsx
        logger.info(f"Результаты baseline сохранены: {output_xlsx}")

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
