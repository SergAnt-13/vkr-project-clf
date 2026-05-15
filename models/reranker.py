"""Cross-encoder reranking for OKPD-2 candidates."""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

from models.retriever import OKPDCandidate

logger = logging.getLogger(__name__)


class SemanticReranker:
    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None

    def load_model(self):
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the semantic reranker. "
                "Install project dependencies from requirements.txt."
            ) from exc

        logger.info("Loading cross-encoder: %s", self.model_name)
        self.model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self.model

    def rerank(self, product_text: str, candidates: Sequence[OKPDCandidate], top_n: int = 3) -> List[OKPDCandidate]:
        if not candidates:
            return []

        model = self.load_model()
        pairs = [(product_text, candidate.description) for candidate in candidates]
        raw_scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = np.asarray(raw_scores, dtype="float32").reshape(-1)
        if len(scores) and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))

        ranked = []
        for candidate, score in zip(candidates, scores):
            ranked.append(
                OKPDCandidate(
                    code=candidate.code,
                    description=candidate.description,
                    retrieval_score=float(score),
                )
            )

        ranked.sort(key=lambda item: item.retrieval_score, reverse=True)
        return ranked[: min(top_n, len(ranked))]
