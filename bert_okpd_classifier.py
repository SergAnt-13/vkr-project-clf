import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import json
import warnings
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

# Отключаем предупреждения о TensorFlow
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class OKPDDataset(Dataset):
    """Датасет для обучения BERT классификатора ОКПД2"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTOKPDClassifier:
    """BERT классификатор для кодов ОКПД2"""

    def __init__(self, config_obj=None, model_name: str = "DeepPavlov/rubert-base-cased"):
        self.config = config_obj
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.test_texts = None
        self.test_labels = None
        self.temperature = 1.0
        self.collect_loss_stats = False
        self.small_loss_threshold = None  # None – Small Loss отключён
        pattern = getattr(self.config, "OKPD2_PATTERN", r"^\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?$")
        self.code_pattern = re.compile(pattern)

        logger.info(f"Инициализация BERT классификатора на устройстве: {self.device}")

    def is_valid_code(self, code) -> bool:
        """Проверить, что код соответствует формату ОКПД2."""
        if not isinstance(code, str):
            return False
        return bool(self.code_pattern.match(code.strip()))

    def prepare_data(self, df: pd.DataFrame, text_column: str = "name_norm",
                     label_column: str = "okpd2_current", min_samples_per_class: int = 10) -> Tuple[
        List[str], List[int]]:
        """Подготовка данных для обучения"""

        # Фильтруем валидные коды
        valid_series = df[label_column].apply(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip().lower() != 'nan' else ""
        )
        valid_mask = valid_series.apply(self.is_valid_code)
        valid_df = df[valid_mask].copy()

        logger.info(f"Найдено {len(valid_df)} записей с валидными кодами ОКПД2")

        # Группируем по кодам и фильтруем редкие классы
        code_counts = valid_df[label_column].value_counts()
        frequent_codes = code_counts[code_counts >= min_samples_per_class].index.tolist()

        # Преобразуем коды в строки для сравнения
        valid_df[label_column] = valid_df[label_column].astype(str)
        frequent_codes_str = [str(code) for code in frequent_codes]

        filtered_df = valid_df[valid_df[label_column].isin(frequent_codes_str)]

        logger.info(
            f"После фильтрации редких классов: {len(filtered_df)} записей, {len(frequent_codes)} уникальных кодов")

        # Создаем маппинг кодов в ID
        unique_codes = sorted([str(code) for code in frequent_codes])
        self.label_to_id = {code: idx for idx, code in enumerate(unique_codes)}
        self.id_to_label = {idx: code for code, idx in self.label_to_id.items()}

        # Подготавливаем данные
        texts = filtered_df[text_column].tolist()
        labels = [self.label_to_id[code] for code in filtered_df[label_column]]

        logger.info(f"Подготовлено {len(texts)} примеров для обучения")

        return texts, labels

    def fit(self, df: pd.DataFrame, text_column: str = "name_norm",
            label_column: str = "okpd2_current", test_size: float = 0.2,
            max_length: int = 128, batch_size: int = 16, num_epochs: int = 5,
            learning_rate: float = 2e-5, min_samples_per_class: int = 10,
            save_path: str = "./bert_model"):
        """Обучение BERT модели"""

        logger.info("🎓 Начинаем обучение BERT классификатора")
        start_time = time.time()

        # Подготовка данных
        texts, labels = self.prepare_data(df, text_column, label_column, min_samples_per_class)

        if len(texts) < 100:
            raise ValueError(f"Недостаточно данных для обучения BERT: {len(texts)} примеров")

        # Инициализация токенизатора и модели
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Увеличиваем таймаут для загрузки
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 минут

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            num_labels = len(self.label_to_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                problem_type="single_label_classification"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.info("Попробуем использовать локальную модель...")

            # Пробуем загрузить из локального кэша
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_path = os.path.join(cache_dir, "models--DeepPavlov--rubert-base-cased")

            if os.path.exists(model_path):
                # Ищем актуальную версию модели
                for item in os.listdir(model_path):
                    if item.startswith("snapshots"):
                        snapshots_dir = os.path.join(model_path, item)
                        if os.path.exists(snapshots_dir):
                            latest_snapshot = max(os.listdir(snapshots_dir))
                            local_model_path = os.path.join(snapshots_dir, latest_snapshot)

                            logger.info(f"Загружаем модель из локального кэша: {local_model_path}")

                            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)

                            num_labels = len(self.label_to_id)
                            self.model = AutoModelForSequenceClassification.from_pretrained(
                                local_model_path,
                                num_labels=num_labels,
                                problem_type="single_label_classification"
                            )
                            break
                else:
                    raise ValueError("Не удалось найти локальную модель")
            else:
                raise ValueError("Локальная модель не найдена")

        self.model.to(self.device)

        logger.info(f"Модель инициализирована с {num_labels} классами")

        # Разделение на train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Сохраняем тестовые данные для последующей оценки иерархии и уверенности
        self.test_texts = test_texts
        self.test_labels = test_labels

        logger.info(f"Разделение данных: train={len(train_texts)}, test={len(test_texts)}")

        # Создание датасетов
        train_dataset = OKPDDataset(train_texts, train_labels, self.tokenizer, max_length)
        test_dataset = OKPDDataset(test_texts, test_labels, self.tokenizer, max_length)

        # Создание DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Настройка оптимизатора
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Обучение
        self.model.train()
        best_f1 = 0.0
        use_small_loss = getattr(self, 'use_small_loss', False)
        loss_values = []

        if use_small_loss:
            logger.info("Small Loss Trick активирован: батчи с высокой потерей будут пропускаться в первой трети эпох")
        patience_counter = 0
        early_stop_patience = 3  # ждём 3 эпохи без улучшений
        avg_loss = 0.0
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Small Loss Trick: пропускаем батчи с аномально высокой потерей на ранних эпохах
                if use_small_loss and epoch < num_epochs // 3:
                    thresh = getattr(self, 'small_loss_threshold', None)
                    if thresh is not None and loss.item() > thresh:
                        continue
                loss.backward()
                optimizer.step()

                if getattr(self, 'collect_loss_stats', False) and epoch < 2:
                    loss_values.append(loss.item())
                total_loss += loss.item()

                # Логируем прогресс каждые 50 батчей
                if batch_idx % 50 == 0:
                    logger.info(
                        f"Эпоха {epoch + 1}/{num_epochs}, Батч {batch_idx}/{len(train_loader)}, Потери: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Эпоха {epoch + 1}/{num_epochs}, Средние потери: {avg_loss:.4f}")

            # Оценка на тестовых данных
            metrics = self._evaluate_model(test_loader)
            logger.info(f"Точность на тестовых данных: {metrics['accuracy']:.3f}")

            # Сохраняем лучшую модель
            if metrics['weighted_f1'] > best_f1:
                best_f1 = metrics['weighted_f1']
                self.save_model(save_path)
                logger.info(f"Новая лучшая модель сохранена (F1 weighted: {best_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"F1 не улучшилась {patience_counter}/{early_stop_patience}")

            if patience_counter >= early_stop_patience:
                logger.info(f"Ранняя остановка на эпохе {epoch + 1}")
                break

        # Финальная оценка
        if getattr(self, 'collect_loss_stats', False) and loss_values:
            arr = np.array(loss_values)
            logger.info(f"Статистика loss (первые 2 эпохи): всего батчей={len(arr)}")
            for p in [50, 75, 90, 95, 99]:
                logger.info(f"  {p}-й процентиль: {np.percentile(arr, p):.4f}")
            logger.info(f"  среднее: {arr.mean():.4f}, std: {arr.std():.4f}")
        final_metrics = self._evaluate_model(test_loader)
        training_time = time.time() - start_time

        logger.info(f"🎯 Итоговые метрики на тесте:")
        logger.info(f"   Точность (Accuracy): {final_metrics['accuracy']:.4f}")
        logger.info(f"   F1 (взвешенная): {final_metrics['weighted_f1']:.4f}")
        logger.info(f"   F1 (макро): {final_metrics['macro_f1']:.4f}")

        self.is_trained = True

        return {
            "accuracy": final_metrics['accuracy'],
            "weighted_f1": final_metrics['weighted_f1'],
            "macro_f1": final_metrics['macro_f1'],
            "classification_report": final_metrics['classification_report'],
            "eval_loss": avg_loss,
            "training_time": training_time,
            "num_classes": num_labels,
            "num_samples": len(texts)
        }

    def _evaluate_model(self, test_loader) -> Dict[str, float]:
        """Расширенная оценка модели с полными метриками"""
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)

        # Важно: список классов берём из label_to_id
        all_class_ids = sorted(self.label_to_id.values())
        all_class_names = [self.id_to_label[c] for c in all_class_ids]

        # Сводка по каждому классу
        report_dict = classification_report(
            all_labels, all_preds,
            labels=all_class_ids,
            target_names=all_class_names,
            output_dict=True,
            zero_division=0
        )

        # Логируем основные агрегированные метрики
        logger.info(f"Точность (Accuracy): {accuracy:.4f}")
        logger.info(f"F1 (weighted): {report_dict['weighted avg']['f1-score']:.4f}")
        logger.info(f"F1 (macro): {report_dict['macro avg']['f1-score']:.4f}")
        logger.info(f"Precision (weighted): {report_dict['weighted avg']['precision']:.4f}")
        logger.info(f"Recall (weighted): {report_dict['weighted avg']['recall']:.4f}")

        # Возвращаем словарь с метриками
        return {
            "accuracy": accuracy,
            "weighted_f1": float(report_dict['weighted avg']['f1-score']),
            "macro_f1": float(report_dict['macro avg']['f1-score']),
            "classification_report": report_dict,
            "y_true": all_labels,
            "y_pred": all_preds
        }

    def set_temperature(self, temperature: float = 2.0):
        """Установить температурный коэффициент для калибровки уверенности."""
        self.temperature = temperature
        logger.info(f"Температура softmax установлена: {self.temperature}")

    def get_test_predictions(self) -> Tuple[List[str], List[str], List[float], List[float]]:
        """Возвращает истинные метки, предсказанные коды, уверенности и энтропии для тестовой выборки."""
        if not self.is_trained or self.test_texts is None:
            raise ValueError("Модель не обучена или тестовые данные отсутствуют.")
        pred_codes, confidences, entropies = self.predict(self.test_texts)
        true_codes = [self.id_to_label[l] for l in self.test_labels]
        return true_codes, pred_codes, confidences, entropies

    def get_entropy(self, logits):
        """Рассчитать энтропию распределения вероятностей Softmax."""
        probs = torch.softmax(logits / getattr(self, 'temperature', 1.0), dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.cpu().tolist()

    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[List[str], List[float], List[float]]:
        """Предсказание кодов ОКПД2. Возвращает списки: предсказанные коды, уверенности, энтропии."""
        if not self.is_trained:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict()")

        self.model.eval()
        predictions = []
        confidences = []
        entropies = []

        dataset = OKPDDataset(texts, [0] * len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                temp = getattr(self, 'temperature', 1.0)
                probs = torch.softmax(logits / temp, dim=-1)
                max_probs, predicted_ids = torch.max(probs, dim=-1)

                batch_entropy = self.get_entropy(logits)

                for pred_id, conf, ent in zip(
                        predicted_ids.cpu().numpy(),
                        max_probs.cpu().numpy(),
                        batch_entropy
                ):
                    if pred_id in self.id_to_label:
                        predictions.append(self.id_to_label[pred_id])
                        confidences.append(float(conf))
                        entropies.append(float(ent))
                    else:
                        predictions.append("")
                        confidences.append(0.0)
                        entropies.append(0.0)

        return predictions, confidences, entropies

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "name_norm") -> pd.DataFrame:
        """Предсказание для DataFrame."""
        texts = df[text_column].tolist()
        predictions, confidences, entropies = self.predict(texts)
        result_df = df.copy()
        result_df['okpd2_pred'] = predictions
        result_df['conf'] = confidences
        result_df['entropy'] = entropies
        return result_df

    def save_model(self, save_path: str = "./bert_model"):
        """Сохранение обученной модели"""

        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не обучена")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Сохранение модели и токенизатора
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Сохранение маппингов
        with open(save_path / "label_mappings.json", "w", encoding="utf-8") as f:
            json.dump({
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Модель сохранена в {save_path}")

    def load_model(self, load_path: str = "./bert_model"):
        """Загрузка обученной модели"""

        load_path = Path(load_path)

        if not load_path.exists():
            raise ValueError(f"Путь к модели не существует: {load_path}")

        # Загрузка токенизатора и модели
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)

        # Загрузка маппингов
        with open(load_path / "label_mappings.json", "r", encoding="utf-8") as f:
            mappings = json.load(f)
            self.label_to_id = mappings["label_to_id"]
            self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}

        self.is_trained = True
        logger.info(f"Модель загружена из {load_path}")

    def get_model_info(self) -> Dict:
        """Получение информации о модели"""

        return {
            "model_name": self.model_name,
            "num_classes": len(self.label_to_id),
            "device": str(self.device),
            "is_trained": self.is_trained,
            "classes": list(self.label_to_id.keys())[:10]  # Первые 10 классов
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Получение распределения классов"""
        return dict(self.label_to_id)
