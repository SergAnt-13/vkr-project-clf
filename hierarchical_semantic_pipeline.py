"""
Иерархический семантический пайплайн: класс → группа → код + тезаурус.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config
from models.reranker import SemanticReranker
from data_loader import DataLoader
from text_processor import TextProcessor
from output_manager import OutputManager
from vat_processor import VATProcessor

logger = logging.getLogger(__name__)


class HierarchicalSemanticPipeline:
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.bi_encoder = SentenceTransformer(config.SEMANTIC_BI_ENCODER_MODEL)
        self.reranker = SemanticReranker(
            model_name=config.SEMANTIC_CROSS_ENCODER_MODEL,
            batch_size=config.SEMANTIC_BATCH_SIZE,
        )
        # Загружаем индексы и ключи
        self.indexes = {}
        self.keys = {}
        for level in ("class", "group", "code"):
            idx_path = config.FAISS_DIR / f"{level}_level" / "index.faiss"
            key_path = config.FAISS_DIR / f"{level}_level" / "keys.txt"
            if idx_path.exists() and key_path.exists():
                self.indexes[level] = faiss.read_index(str(idx_path))
                with open(key_path, encoding="utf-8") as f:
                    self.keys[level] = [line.strip() for line in f if line.strip()]
            else:
                raise FileNotFoundError(f"Индекс {level}_level не найден, запустите build_hierarchical_index.py")

        # Попытка загрузить тезаурус (если есть)
        thesaurus_path = Path("food_thesaurus.xlsx")
        self.thesaurus = {}
        if thesaurus_path.exists():
            try:
                df = pd.read_excel(thesaurus_path)
                for _, row in df.iterrows():
                    term = str(row['term']).strip().lower()
                    hyper = str(row['hypernyms']).strip()
                    if term and hyper:
                        self.thesaurus[term] = hyper.split("; ")
            except Exception:
                logger.warning("Не удалось загрузить тезаурус")

    def search_level(self, level: str, query: str, k: int = 5) -> List[tuple]:
        vec = self.bi_encoder.encode([query])
        faiss.normalize_L2(vec)
        dist, ids = self.indexes[level].search(vec, k)
        return [(self.keys[level][i], float(dist[0][j])) for j, i in enumerate(ids[0])]

    def enrich_query(self, text: str) -> str:
        """Добавляет гиперонимы из тезауруса, если есть."""
        if not self.thesaurus:
            return text
        tokens = text.lower().split()
        additions = []
        for token in tokens:
            if token in self.thesaurus:
                additions.extend(self.thesaurus[token])
        if additions:
            return text + " " + " ".join(additions[:3])  # не более 3 гиперонимов
        return text

    def classify_product(self, product_text: str) -> Dict:
        # Шаг 1: класс — топ‑3 для веса
        class_candidates = self.search_level("class", product_text, k=3)
        best_class = class_candidates[0][0] if class_candidates else ""

        # Шаг 2: группа — топ‑5
        group_candidates = self.search_level("group", product_text, k=5)
        best_group = group_candidates[0][0] if group_candidates else ""

        # Шаг 3: коды — топ‑50 (без фильтрации)
        code_candidates = self.search_level("code", product_text, k=20)
        if not code_candidates:
            if group_candidates:
                return {"predicted_code": group_candidates[0][0], "score": group_candidates[0][1],
                        "flag": "Сомнительный"}
            if class_candidates:
                return {"predicted_code": class_candidates[0][0], "score": class_candidates[0][1],
                        "flag": "Неопознанный"}
            return {"predicted_code": "", "score": 0.0, "flag": "Неопознанный"}

        # === ДИВЕРСИФИКАЦИЯ ПО ГРУППАМ ===
        # Группируем коды по первым 5 цифрам, берём лучший из каждой группы
        group_best = {}
        for code, score in code_candidates:
            grp = code[:5] if len(code) >= 5 else code
            if grp not in group_best or score > group_best[grp][1]:
                group_best[grp] = (code, score)

        # Сортируем группы по скору лучшего кода, берём топ‑10 групп
        sorted_groups = sorted(group_best.items(), key=lambda x: x[1][1], reverse=True)[:10]
        diverse_candidates = [(code, score) for _, (code, score) in sorted_groups]

        # Добавляем иерархические бонусы
        boosted = []
        for code, bi_score in diverse_candidates:
            bonus = 0.0
            if best_class and code.startswith(best_class):
                bonus += 0.05
            if best_group and code.startswith(best_group):
                bonus += 0.10
            boosted.append((code, bi_score + bonus))

        # Сортируем по boosted-скору и берём топ‑5 для Cross-Encoder
        boosted.sort(key=lambda x: x[1], reverse=True)
        top_codes = boosted[:5]

        # Без Cross-Encoder: просто берём лучший boosted-код
        top_code, top_score = boosted[0] if boosted else ("", 0.0)

        # Cross-Encoder на топ‑5
        # class DummyCandidate:
        #     def __init__(self, code, desc):
        #         self.code = code
        #         self.description = desc
        # candidate_objs = [DummyCandidate(c, self._get_code_desc(c)) for c, _ in top_codes]
        # ranked = self.reranker.rerank(product_text, candidate_objs, top_n=1)
        #
        # if ranked:
        #     top_code = ranked[0].code
        #     top_score = ranked[0].retrieval_score
        # else:
        #     top_code = top_codes[0][0]
        #     top_score = top_codes[0][1]

        # Fallback на группу, если уверенность низкая
        if top_score < 0.4 and best_group:
            top_code = best_group
            top_score = group_candidates[0][1]

        flag = "Уверен" if top_score > 0.7 else ("Сомнительный" if top_score > 0.4 else "Неопознанный")
        return {"predicted_code": top_code, "score": top_score, "flag": flag}

    def _get_code_desc(self, code: str) -> str:
        """Загружает описание кода из классификатора (упрощённо)."""
        ref = pd.read_excel(config.OKPD2_REFERENCE_FILE)
        code_col = next((c for c in ref.columns if str(c).strip().lower() in ('code', 'okpd', 'okpd2')), None)
        desc_col = next((c for c in ref.columns if str(c).strip().lower() in ('description', 'desc', 'name')), None)
        if code_col and desc_col:
            row = ref[ref[code_col].astype(str).str.strip() == code]
            if not row.empty:
                return str(row.iloc[0][desc_col])
        return code