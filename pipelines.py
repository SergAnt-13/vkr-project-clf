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
        training_file: Optional[Path] = None,
    ):
        runtime = replace(self.config)
        # Если указан training_file, создаём подпапку с его именем
        if training_file:
            base_dir = Path(output_dir) if output_dir else runtime.BERT_OUTPUT_DIR
            runtime.OUTPUT_DIR = base_dir / Path(training_file).stem
        else:
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
        runtime = self._runtime_config(output_dir, model_dir, training_file)
        loader = DataLoader(runtime)
        vat_processor = VATProcessor(runtime)
        output_manager = OutputManager(runtime)
        classifier = BERTOKPDClassifier(runtime, model_name=runtime.BERT_MODEL_NAME)


        # Если prediction_file передан явно – используем его,
        # иначе для обычного режима всегда берём основной файл (to_process)
        if prediction_file:
            prediction_path = Path(prediction_file)
        else:
            prediction_path = runtime.PRODUCTS_FILE  # <-- ~52К записей
        prediction_raw = loader.read_table(prediction_path)
        prediction_df = loader.standardize_products(prediction_raw, max_rows=max_rows)

        prediction_df["okpd2_current"] = prediction_raw.get("Код ОКПД2", pd.Series(dtype=str)).astype(str).str.strip()
        prediction_df["vat_current"] = prediction_raw.get("Ставка НДС (Заказ)", pd.Series(dtype=str)).astype(str).str.strip()

        training_df, training_path, label_column = self._resolve_training_source(
            loader=loader,
            mode=mode,
            training_file=training_file,
            prediction_path=prediction_path,
            training_max_rows=training_max_rows,
        )

        abbreviation_rules = loader.load_abbreviations()
        self.abbreviation_rules = abbreviation_rules
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

        # Вычисляем prior на основе обучающей выборки
        if 'training_label' in training_df.columns:
            prior = training_df['training_label'].value_counts(normalize=True).to_dict()
        else:
            prior = training_df['okpd2_current'].value_counts(normalize=True).to_dict()
        # Сохраняем в классификатор (если он ожидает)
        classifier.prior = prior

        classifier.set_temperature(1.0)
        predictions_df = classifier.predict_dataframe(prediction_df, text_column="name_norm")

        # === Коррекция уверенности на основе prior (частот классов в обучении) ===
        if hasattr(classifier, 'prior'):
            max_prior = max(classifier.prior.values())

            def adjust_conf(row):
                code = row['okpd2_pred']
                if code in classifier.prior:
                    prior_weight = classifier.prior[code] / max_prior
                    return row['conf'] * prior_weight
                return row['conf']

            predictions_df['conf'] = predictions_df.apply(adjust_conf, axis=1)
            logger.info("Уверенность скорректирована на основе априорных вероятностей классов")

        # === Применяем экспертные правила (доменный бустинг) ===
        predictions_df = self.apply_domain_boosting(predictions_df, text_column="name_norm")
        predictions_df["okpd2_final"] = predictions_df.apply(self._choose_final_okpd, axis=1)

        pp908_text = loader.load_pp908_text()
        if pp908_text:
            vat_processor.build_vat10_prefixes(pp908_text)

        predictions_df["vat_pred"] = vat_processor.process_vat_predictions(predictions_df, "okpd2_final")
        predictions_df["vat_final"] = predictions_df.apply(self._choose_final_vat, axis=1)

        # === Иерархическая точность на основе исходных кодов (если есть) ===
        if "okpd2_current" in predictions_df.columns:
            y_true = predictions_df["okpd2_current"].astype(str).str.strip()
            y_pred = predictions_df["okpd2_pred"].astype(str).str.strip()

            def hierarchical_accuracy_prod(y_true, y_pred, level):
                correct = total = 0
                for t, p in zip(y_true, y_pred):
                    # Приводим к строке и отбрасываем пустые/NaN
                    t_str = str(t).strip() if pd.notna(t) else ""
                    p_str = str(p).strip() if pd.notna(p) else ""
                    if t_str in ("", "nan", "unknown") or p_str in ("", "nan", "unknown"):
                        continue
                    if level == 1 and t_str[:2] == p_str[:2]:
                        correct += 1
                    elif level == 2 and len(t_str) >= 5 and len(p_str) >= 5 and t_str[:5] == p_str[:5]:
                        correct += 1
                    elif level == 3 and t_str == p_str:
                        correct += 1
                    total += 1
                return correct / total if total > 0 else 0.0

            acc_class = hierarchical_accuracy_prod(y_true, y_pred, level=1)
            acc_group = hierarchical_accuracy_prod(y_true, y_pred, level=2)
            acc_full  = hierarchical_accuracy_prod(y_true, y_pred, level=3)

            logger.info(f"Иерархическая точность на всём файле: класс={acc_class:.4f}, группа={acc_group:.4f}, полный код={acc_full:.4f}")

            # Сохраним эти метрики в отчёт
            self._prod_metrics = {
                "acc_class": acc_class,
                "acc_group": acc_group,
                "acc_full": acc_full,
            }
        else:
            self._prod_metrics = {}

        # === Формирование таблиц анализа (как в Baseline) ===
        if "okpd2_current" in predictions_df.columns and "vat_current" in predictions_df.columns:
            predictions_df["okpd2_final"] = predictions_df.apply(self._choose_final_okpd, axis=1)
            tables = output_manager.create_analysis_tables(predictions_df)
            output_manager.save_tables(tables, predictions_df)
        else:
            logger.info("Нет эталонных кодов/ставок для сравнения — таблицы неверных не созданы.")

        # ========== PRODUCTION ВИЗУАЛИЗАЦИЯ ==========
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # 1. Гистограмма уверенности
        plt.figure(figsize=(10, 6))
        plt.hist(predictions_df['conf'], bins=50, alpha=0.7, color='steelblue')
        plt.title('Распределение уверенности модели (Production 52K)')
        plt.xlabel('Уверенность')
        plt.ylabel('Количество')
        plt.axvline(x=0.9, color='red', linestyle='--', label='Порог 0.9')
        plt.legend()
        prod_conf_path = runtime.OUTPUT_DIR / 'confidence_distribution_prod.png'
        plt.savefig(prod_conf_path, dpi=150)
        plt.close()

        # 2. Топ-20 предсказанных кодов
        top20 = predictions_df['okpd2_pred'].value_counts().nlargest(20)
        plt.figure(figsize=(12, 6))
        top20.plot(kind='bar')
        plt.title('Топ-20 предсказанных кодов ОКПД2 (Production 52K)')
        plt.xlabel('Код ОКПД2')
        plt.ylabel('Количество')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        prod_codes_path = runtime.OUTPUT_DIR / 'predicted_codes_top20.png'
        plt.savefig(prod_codes_path, dpi=150)
        plt.close()

        # 3. Сравнение длин названий (train vs prod)
        if hasattr(self, '_prod_metrics'):  # у нас есть данные из обучающей выборки?
            # Длины train можно взять из training_df (он доступен)
            train_lengths = training_df['name_norm'].str.len().dropna()
            prod_lengths = prediction_df['name_norm'].str.len().dropna()
            plt.figure(figsize=(10, 6))
            plt.hist(train_lengths, bins=30, alpha=0.5, label='Train (enriched)', color='blue')
            plt.hist(prod_lengths, bins=30, alpha=0.5, label='Production (52K)', color='orange')
            plt.title('Распределение длин названий товаров')
            plt.xlabel('Длина (символов)')
            plt.ylabel('Плотность')
            plt.legend()
            drift_path = runtime.OUTPUT_DIR / 'name_length_drift.png'
            plt.savefig(drift_path, dpi=150)
            plt.close()

        # 4. Иерархическая точность (bar chart) – если сохранены метрики
        if hasattr(self, '_prod_metrics') and self._prod_metrics:
            prod_acc = [self._prod_metrics['acc_class'], self._prod_metrics['acc_group'], self._prod_metrics['acc_full']]
            # Для test-метрик надо бы сохранить, но пока нет. Можно опустить.
            # Сделаем только prod
            fig, ax = plt.subplots(figsize=(8, 5))
            levels = ['Класс (XX)', 'Группа (XX.XX)', 'Полный код']
            ax.bar(levels, prod_acc, color=['green', 'orange', 'red'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Точность')
            ax.set_title('Иерархическая точность на Production-файле')
            for i, v in enumerate(prod_acc):
                ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
            hier_path = runtime.OUTPUT_DIR / 'hierarchical_accuracy_prod.png'
            plt.savefig(hier_path, dpi=150)
            plt.close()
            logger.info(f"Графики сохранены: {prod_conf_path}, {prod_codes_path}, {drift_path}, {hier_path}")
        else:
            logger.info(f"Графики сохранены: {prod_conf_path}, {prod_codes_path}, {drift_path}")

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
            if value and value.lower() not in ('', 'nan', 'ндс0', 'ндс12', 'ндс18', 'ндс20'):
                if value == 'НДС20':
                    return 'НДС22'
                return value
        return 'НДС22'

    def apply_domain_boosting(self, predictions_df: pd.DataFrame, text_column: str = "name_norm") -> pd.DataFrame:
        if not hasattr(self, 'abbreviation_rules'):
            logger.warning("Правила сокращений не загружены, доменный бустинг пропущен")
            return predictions_df

        boosted = 0
        corrected = 0
        for idx, row in predictions_df.iterrows():
            text = str(row.get(text_column, '')).lower()
            for rule in self.abbreviation_rules:
                keyword = rule['replacement'].lower()
                okpd_codes_str = rule.get('okpd_codes', '')
                if not keyword or not okpd_codes_str:
                    continue
                if keyword in text:
                    possible_codes = [c.strip() for c in okpd_codes_str.split(';') if c.strip()]
                    if not possible_codes:
                        continue
                    if row['okpd2_pred'] in possible_codes:
                        predictions_df.at[idx, 'conf'] = 0.99
                        boosted += 1
                    elif row['conf'] < 0.5:
                        predictions_df.at[idx, 'okpd2_pred'] = possible_codes[0]
                        predictions_df.at[idx, 'conf'] = 0.9
                        corrected += 1
                    break
        logger.info(f"Доменный бустинг: повышена уверенность у {boosted} записей, исправлен код у {corrected}")
        return predictions_df

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

        if hasattr(self, '_prod_metrics') and self._prod_metrics:
            lines.append("")
            lines.append("Иерархическая точность на Production-файле:")
            lines.append(f"- Класс (XX): {self._prod_metrics['acc_class']:.4f}")
            lines.append(f"- Группа (XX.XX): {self._prod_metrics['acc_group']:.4f}")
            lines.append(f"- Полный код: {self._prod_metrics['acc_full']:.4f}")

        if metrics:
            lines.append("")
            lines.append("Метрики качества классификации:")
            lines.append(f"- Accuracy (Точность): {metrics.get('accuracy', 0.0):.4f}")
            lines.append(f"- F1 (взвешенная): {metrics.get('weighted_f1', 0.0):.4f}")
            lines.append(f"- F1 (макро): {metrics.get('macro_f1', 0.0):.4f}")
            # classification_report при желании можно не выводить, иначе очень много строк
        return "\n".join(lines)