"""BERT OKPD-2 classification pipeline."""

from dataclasses import replace
import logging
from pathlib import Path
from typing import Dict, Optional

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from bert_okpd_classifier import BERTOKPDClassifier
from config import config
from data_loader import DataLoader
from output_manager import OutputManager
from text_processor import TextProcessor
from vat_processor import VATProcessor

logger = logging.getLogger(__name__)


def _valid_okpd_mask(series: pd.Series, pattern: str) -> pd.Series:
    return series.astype(str).str.strip().str.match(pattern).fillna(False)


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
            epochs: Optional[int] = None
    ) -> Dict[str, object]:
        runtime = self._runtime_config(output_dir, model_dir, training_file)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        runtime.OUTPUT_DIR = runtime.OUTPUT_DIR / timestamp
        runtime.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        prediction_df["vat_current"] = prediction_raw.get("Ставка НДС (Заказ)", pd.Series(dtype=str)).astype(
            str).str.strip()

        training_df, training_path, label_column = self._resolve_training_source(
            loader=loader,
            mode=mode,
            training_file=training_file,
            prediction_path=prediction_path,
            training_max_rows=training_max_rows,
        )

        abbreviation_rules = loader.load_abbreviations()
        abbreviation_rules = loader.load_abbreviations()
        logger.info(f"Загружено правил: {len(abbreviation_rules)}")
        for rule in abbreviation_rules[:5]:
            logger.info(
                f"  pattern='{rule['pattern']}', replacement='{rule['replacement']}', okpd_codes='{rule['okpd_codes']}'")
        self.abbreviation_rules = abbreviation_rules
        text_processor = TextProcessor(abbreviation_rules=abbreviation_rules, config_obj=runtime)

        prediction_df["name_norm"] = text_processor.normalize_text_series(prediction_df["name_raw"])
        training_df["name_norm"] = text_processor.normalize_text_series(training_df["name_raw"])
        training_df = training_df[training_df["name_norm"].astype(str).str.len() > 0].copy()

        if validate:
            # Честная валидация через внутренний split BERT
            fit_kwargs = self._bert_fit_kwargs(runtime, mode)
            if epochs is not None:
                fit_kwargs["num_epochs"] = epochs
                logger.info(f"Число эпох переопределено через CLI: {epochs}")
            # classifier.use_small_loss = True
            # classifier.small_loss_threshold = 3.86
            # classifier.collect_loss_stats = True
            metrics = classifier.fit(
                training_df,
                text_column="name_norm",
                label_column=label_column,
                save_path=str(runtime.MODEL_DIR) if not load_existing_model else None,
                num_epochs=epochs if epochs is not None else fit_kwargs.get("num_epochs", 5),
                **{k: v for k, v in fit_kwargs.items() if k != "num_epochs"},
            )

            # Получаем предсказания на тестовой выборке, которую BERT сохранил внутри
            y_true, y_pred_str, confidences, entropies = classifier.get_test_predictions()
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
            # classifier.use_small_loss = True
            # classifier.small_loss_threshold = 3.86
        else:
            fit_kwargs = self._bert_fit_kwargs(runtime, mode)
            if epochs is not None:
                fit_kwargs["num_epochs"] = epochs
                logger.info(
                    f"DEBUG: epochs={epochs}, num_epochs будет: {epochs if epochs is not None else fit_kwargs.get('num_epochs', 5)}")
            # classifier.collect_loss_stats = True
            bert_metrics = classifier.fit(
                training_df,
                text_column="name_norm",
                label_column=label_column,
                save_path=str(runtime.MODEL_DIR),
                num_epochs=epochs if epochs is not None else fit_kwargs.get("num_epochs", 5),
                **{k: v for k, v in fit_kwargs.items() if k != "num_epochs"},
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

        # Добавляем правила из классификатора и валидируем
        self._add_classifier_rules()
        predictions_df = self._validate_and_fix_codes(predictions_df)

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
            acc_full = hierarchical_accuracy_prod(y_true, y_pred, level=3)

            logger.info(
                f"Иерархическая точность на всём файле: класс={acc_class:.4f}, группа={acc_group:.4f}, полный код={acc_full:.4f}")

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
            prod_acc = [self._prod_metrics['acc_class'], self._prod_metrics['acc_group'],
                        self._prod_metrics['acc_full']]
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
        """
        Повысить уверенность и скорректировать предсказания на основе ключевых слов из сокращений
        и связанных с ними кодов ОКПД2.
        """
        import pandas as pd
        abbrev_path = self.config.ABBREVIATIONS_FILE
        if not abbrev_path.exists():
            logger.warning(f"Файл сокращений не найден: {abbrev_path}")
            return predictions_df

        df = pd.read_excel(abbrev_path)
        # Ищем колонку с кодами
        okpd_col = None
        for name in ['okpd_codes', 'okpd', 'codes']:
            if name in df.columns:
                okpd_col = name
                break
        if okpd_col is None:
            okpd_col = df.columns[-1]  # последняя колонка

        # Собираем правила в словарь: расшифровка -> список кодов
        rules_dict = {}
        for _, row in df.iterrows():
            keyword = str(row.get('expansion', '')).strip().lower()
            okpd_codes_str = str(row[okpd_col]).strip()
            if not keyword or not okpd_codes_str or okpd_codes_str.lower() == 'nan':
                continue
            possible_codes = [c.strip() for c in okpd_codes_str.split(';') if c.strip()]
            if possible_codes:
                rules_dict[keyword] = possible_codes

        logger.info(f"Загружено {len(rules_dict)} правил для доменного бустинга")
        # Отладочный вывод первых 3 правил
        sample = list(rules_dict.items())[:3]
        for kw, codes in sample:
            logger.info(f"  '{kw}' -> {codes}")

        boosted = 0
        corrected = 0
        for idx, row in predictions_df.iterrows():
            text = str(row.get(text_column, '')).lower()
            for keyword, possible_codes in rules_dict.items():
                if keyword in text:
                    if row['okpd2_pred'] in possible_codes:
                        predictions_df.at[idx, 'conf'] = 0.99
                        boosted += 1
                    elif row['conf'] < 0.5:
                        predictions_df.at[idx, 'okpd2_pred'] = possible_codes[0]
                        predictions_df.at[idx, 'conf'] = 0.9
                        corrected += 1
                    break  # только первое совпадение
        logger.info(f"Доменный бустинг: повышена уверенность у {boosted} записей, исправлен код у {corrected}")
        return predictions_df

    def _add_classifier_rules(self):
        """Обогащает правила доменного бустинга ключевыми словами из классификатора ОКПД-2."""
        ref_path = self.config.OKPD2_REFERENCE_FILE
        if not ref_path.exists():
            logger.info(f"Классификатор {ref_path} не найден, пропускаем.")
            return

        ref_df = pd.read_excel(ref_path)
        # Ищем колонки с кодом и описанием
        code_col = None
        desc_col = None
        for col in ref_df.columns:
            col_lower = col.strip().lower()
            if col_lower in ('code', 'okpd', 'okpd2'): code_col = col
            if col_lower in ('description', 'desc', 'name', 'title'): desc_col = col

        if code_col is None or desc_col is None:
            logger.warning("Не удалось найти колонки code/description в классификаторе.")
            return

        rules_dict = {}
        for _, row in ref_df.iterrows():
            code = str(row[code_col]).strip()
            desc = str(row[desc_col]).lower().strip()
            # Берём только осмысленные слова из описания (длиннее 3 букв)
            keywords = [w for w in re.findall(r'[а-яё]+', desc) if len(w) > 3]
            for kw in keywords:
                if kw not in rules_dict:
                    rules_dict[kw] = []
                if code not in rules_dict[kw]:
                    rules_dict[kw].append(code)

        # Добавляем к существующим правилам бустинга
        if not hasattr(self, 'boosting_rules'):
            self.boosting_rules = {}
        self.boosting_rules.update(rules_dict)
        logger.info(f"Добавлено {len(rules_dict)} правил из классификатора в доменный бустинг.")

    def _validate_and_fix_codes(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверяет, что предсказанные коды есть в официальном классификаторе ОКПД2.
        Если код невалидный — заменяет на ближайший родительский код или помечает флагом.
        """
        ref_path = self.config.OKPD2_REFERENCE_FILE
        if not ref_path.exists():
            logger.warning("Классификатор для валидации не найден.")
            return predictions_df

        ref_df = pd.read_excel(ref_path)
        code_col = next((c for c in ref_df.columns if str(c).strip().lower() in ('code', 'okpd', 'okpd2')), None)
        if code_col is None:
            logger.warning("Колонка code не найдена в классификаторе.")
            return predictions_df

        valid_codes = set(ref_df[code_col].astype(str).str.strip())

        # Строим словарь родительских кодов (если есть parent_code)
        parent_map = {}
        if 'parent_code' in ref_df.columns:
            for _, row in ref_df.iterrows():
                code = str(row[code_col]).strip()
                parent = str(row['parent_code']).strip()
                if parent and parent != 'nan':
                    parent_map[code] = parent

        fixed = 0
        for idx, row in predictions_df.iterrows():
            code = str(row.get('okpd2_pred', '')).strip()
            if not code or code in valid_codes:
                continue

            # Пытаемся найти ближайший валидный родительский код
            current = code
            while current in parent_map:
                current = parent_map[current]
                if current in valid_codes:
                    predictions_df.at[idx, 'okpd2_pred'] = current
                    predictions_df.at[idx, 'conf'] = max(0.3, float(row.get('conf', 0.0)))
                    fixed += 1
                    break
            else:
                # Если не нашли валидного предка — помечаем как невалидный
                predictions_df.at[idx, 'okpd2_pred'] = ""
                predictions_df.at[idx, 'conf'] = 0.0
                fixed += 1

        if fixed:
            logger.info(f"Исправлено/помечено невалидных кодов: {fixed}")
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

        # Accuracy НДС (сравнение с исходными ставками в файле)
        if "vat_current" in predictions_df.columns and "vat_final" in predictions_df.columns:
            vat_matches = (predictions_df["vat_current"] == predictions_df["vat_final"]).sum()
            vat_total = predictions_df["vat_current"].notna().sum()
            vat_accuracy = vat_matches / vat_total if vat_total > 0 else 0.0
            lines.append("")
            lines.append("Accuracy НДС (совпадение с исходными ставками):")
            lines.append(f"- Совпало: {vat_matches} из {vat_total} ({vat_accuracy:.4f})")

        # Топ-20 кодов по средней уверенности (текстовый блок)
        if "okpd2_pred" in predictions_df.columns and "conf" in predictions_df.columns:
            class_confidence = predictions_df.groupby("okpd2_pred")["conf"].agg(["mean", "count"]).reset_index()
            class_confidence.columns = ["code", "avg_conf", "count"]
            class_confidence = class_confidence.sort_values("avg_conf", ascending=False).head(20)
            lines.append("")
            lines.append("Топ-20 кодов по средней уверенности:")
            for _, row in class_confidence.iterrows():
                lines.append(f"  {row['code']}: ср.уверенность={row['avg_conf']:.4f}, записей={int(row['count'])}")

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
