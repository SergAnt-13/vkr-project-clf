"""
Модуль для классификации товаров по кодам ОКПД2
"""
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import config
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)


class OKPDClassifier:
    """Классификатор для предсказания кодов ОКПД2"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.vectorizer = None
        self.labeled_data = None
        self.labeled_codes = None
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компилировать регулярные выражения"""
        self.code_pattern = re.compile(self.config.OKPD2_PATTERN)
    
    def is_valid_code(self, code) -> bool:
        """Проверить, является ли код валидным ОКПД2"""
        if not isinstance(code, str):
            return False
        return bool(self.code_pattern.match(code.strip()))
    
    def split_labeled_unlabeled(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Разделить данные на размеченные и неразмеченные"""
        df["okpd2_current"] = df["okpd2_current"].apply(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip().lower() != 'nan' else ""
        )
        labeled_mask = df["okpd2_current"].apply(self.is_valid_code)
        labeled = df[labeled_mask].copy()
        unlabeled = df[~labeled_mask].copy()
        logger.info(f"Размеченных записей: {len(labeled)}, неразмеченных: {len(unlabeled)}")
        return labeled, unlabeled
    
    def fit(self, labeled_df: pd.DataFrame, text_column: str = "name_norm"):
        """Обучить классификатор на размеченных данных"""
        if len(labeled_df) < self.config.MIN_LABELED_SAMPLES:
            raise ValueError(f"Недостаточно размеченных данных: {len(labeled_df)} < {self.config.MIN_LABELED_SAMPLES}")
        
        logger.info(f"Обучение классификатора на {len(labeled_df)} образцах")
        
        # Проверка на пустые тексты
        non_empty_texts = labeled_df[text_column].str.strip().str.len() > 0
        if not non_empty_texts.any():
            raise ValueError("Все тексты пустые после нормализации")
        
        # Фильтрация пустых текстов
        valid_labeled = labeled_df[non_empty_texts].copy()
        if len(valid_labeled) < self.config.MIN_LABELED_SAMPLES:
            raise ValueError(f"Недостаточно непустых текстов: {len(valid_labeled)} < {self.config.MIN_LABELED_SAMPLES}")
        
        logger.info(f"Используется {len(valid_labeled)} записей с непустыми текстами")
        
        # Инициализация векторайзера
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            min_df=self.config.TFIDF_MIN_DF,
            max_features=self.config.TFIDF_MAX_FEATURES,
        )
        
        try:
            # Обучение векторайзера и трансформация данных
            self.labeled_data = self.vectorizer.fit_transform(valid_labeled[text_column])
            self.labeled_codes = valid_labeled["okpd2_current"].tolist()
            
            logger.info("Обучение классификатора завершено")
            
        except ValueError as e:
            if "empty vocabulary" in str(e):
                logger.error("Ошибка: пустой словарь после векторизации. Возможно, все тексты слишком короткие или содержат только стоп-слова")
                raise ValueError("Не удается создать словарь признаков из текстов")
            else:
                raise
    
    def predict(self, unlabeled_df: pd.DataFrame, text_column: str = "name_norm") -> pd.DataFrame:
        """Предсказать коды ОКПД2 для неразмеченных данных"""
        if self.vectorizer is None or self.labeled_data is None:
            raise ValueError("Классификатор не обучен. Вызовите fit() перед predict()")
        
        if len(unlabeled_df) == 0:
            return pd.DataFrame({"item_id": [], "okpd2_pred": [], "conf": []})
        
        logger.info(f"Предсказание кодов для {len(unlabeled_df)} образцов")
        
        # Трансформация неразмеченных данных
        unlabeled_vectors = self.vectorizer.transform(unlabeled_df[text_column])
        
        # Вычисление сходства
        similarities = cosine_similarity(unlabeled_vectors, self.labeled_data, dense_output=False)
        
        # Генерация предсказаний
        predictions = []
        confidences = []
        
        for i in range(similarities.shape[0]):
            pred, conf = self._predict_single(similarities.getrow(i))
            predictions.append(pred)
            confidences.append(conf)
        
        result_df = pd.DataFrame({
            "item_id": unlabeled_df["item_id"],
            "okpd2_pred": predictions,
            "conf": confidences
        })
        
        valid_predictions = sum(1 for p in predictions if p != "")
        logger.info(f"Получено {valid_predictions} валидных предсказаний из {len(predictions)}")
        
        return result_df
    
    def _predict_single(self, similarity_row) -> Tuple[str, float]:
        """Предсказать код для одного образца"""
        if similarity_row.nnz == 0:
            return "", 0.0
        
        # Получить топ-k наиболее похожих образцов
        k = min(self.config.KNN_K, similarity_row.nnz)
        top_indices = np.argsort(similarity_row.data)[-k:][::-1]
        top_similarities = similarity_row.data[top_indices]
        top_labeled_indices = similarity_row.indices[top_indices]
        
        # Голосование с весами
        vote_scores = {}
        for idx, sim in zip(top_labeled_indices, top_similarities):
            code = self.labeled_codes[idx]
            if self.is_valid_code(code):
                vote_scores[code] = vote_scores.get(code, 0.0) + float(sim)
        
        if not vote_scores:
            return "", 0.0
        
        # Выбор лучшего кода
        best_code, best_score = max(vote_scores.items(), key=lambda x: x[1])
        
        # Вычисление уверенности
        total_score = float(top_similarities.sum()) + 1e-9
        confidence = best_score / total_score
        top_similarity = float(top_similarities.max()) if len(top_similarities) > 0 else 0.0
        
        # Проверка пороговых значений
        if confidence >= self.config.MIN_CONFIDENCE and top_similarity >= self.config.MIN_TOP_SIMILARITY:
            return best_code, confidence
        else:
            return "", confidence
    
    def get_prediction_statistics(self, predictions_df: pd.DataFrame) -> dict:
        """Получить статистику по предсказаниям"""
        total = len(predictions_df)
        valid = (predictions_df["okpd2_pred"] != "").sum()
        high_conf = (predictions_df["conf"] >= self.config.MIN_CONFIDENCE).sum()
        
        stats = {
            'total_predictions': total,
            'valid_predictions': valid,
            'high_confidence_predictions': high_conf,
            'prediction_rate': valid / total if total > 0 else 0,
            'average_confidence': predictions_df["conf"].mean() if total > 0 else 0.0,
            'max_confidence': predictions_df["conf"].max() if total > 0 else 0.0,
            'min_confidence': predictions_df["conf"].min() if total > 0 else 0.0,
        }
        
        return stats

    def evaluate(self, test_df: pd.DataFrame, text_column: str = "name_norm") -> dict:
        """
        Оценить качество классификатора на тестовых данных.

        Parameters:
            test_df: DataFrame с колонками text_column и "okpd2_current" (истинные коды)
            text_column: название колонки с нормализованными текстами

        Returns:
            словарь с метриками accuracy, f1_macro, f1_weighted и classification_report
        """
        if self.vectorizer is None or self.labeled_data is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        # Предсказание на тестовых данных
        predictions_df = self.predict(test_df, text_column=text_column)

        # Извлекаем истинные и предсказанные значения
        y_true = test_df["okpd2_current"].astype(str).str.strip()
        y_pred = predictions_df["okpd2_pred"].astype(str).str.strip()

        # Заменяем пустые предсказания на "unknown" для корректного отчёта
        # (можно оставить как есть, но тогда пустые строки будут считаться отдельным классом)
        y_pred = y_pred.replace("", "unknown")

        # Вычисляем метрики
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Логируем ключевые метрики
        logger.info("=== Оценка качества Baseline-модели ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 (macro): {f1_macro:.4f}")
        logger.info(f"F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"Precision (weighted): {report['weighted avg']['precision']:.4f}")
        logger.info(f"Recall (weighted): {report['weighted avg']['recall']:.4f}")

        # Формируем возвращаемый словарь
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report
        }

        return metrics