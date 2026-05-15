"""Bi-encoder retrieval over OKPD-2 descriptions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OKPDCandidate:
    code: str
    description: str
    retrieval_score: float


class SemanticRetriever:
    """Vector search for official OKPD-2 descriptions.

    FAISS is used when available. A numpy inner-product fallback keeps tests and
    lightweight local checks runnable in environments where faiss-cpu is absent.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        finetuned_dir: Path,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.finetuned_dir = Path(finetuned_dir)
        self.batch_size = batch_size
        self.model = None
        self.codes: List[str] = []
        self.descriptions: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.index = None
        self.faiss = None

    def load_model(self):
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the semantic retriever. "
                "Install project dependencies from requirements.txt."
            ) from exc

        model_path = self.finetuned_dir if self.finetuned_dir.exists() else self.model_name
        logger.info("Loading bi-encoder: %s", model_path)
        self.model = SentenceTransformer(str(model_path))
        return self.model

    def maybe_finetune(self, training_pairs: Sequence[tuple[str, str]], epochs: int = 5, lr: float = 2e-5) -> None:
        """Fine-tune the bi-encoder if a saved tuned model does not exist.

        This is intentionally conservative: if optional training dependencies or
        usable pairs are missing, retrieval still works with the base model.
        """
        if self.finetuned_dir.exists() or len(training_pairs) < 2:
            return

        try:
            from sentence_transformers import InputExample, losses
            from torch.utils.data import DataLoader
        except ImportError:
            logger.warning("Skipping bi-encoder fine-tuning: training dependencies are unavailable.")
            return

        model = self.load_model()
        examples = [InputExample(texts=[left, right]) for left, right in training_pairs if left and right]
        if len(examples) < 2:
            return

        logger.info("Fine-tuning bi-encoder on %s product-description pairs", len(examples))
        train_loader = DataLoader(examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=epochs,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
        )
        self.finetuned_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.finetuned_dir))
        self.model = model

    def build_or_load(self, reference_df: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.cache_dir / "okpd2_metadata.json"
        embeddings_path = self.cache_dir / "okpd2_embeddings.npy"
        faiss_path = self.cache_dir / "okpd2.index"

        self._try_import_faiss()

        if metadata_path.exists() and embeddings_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.codes = metadata["codes"]
            self.descriptions = metadata["descriptions"]
            self.embeddings = np.load(embeddings_path).astype("float32")
            if self.faiss is not None and faiss_path.exists():
                self.index = self.faiss.read_index(str(faiss_path))
            else:
                self._build_numpy_index()
            logger.info("Loaded semantic index with %s OKPD-2 descriptions", len(self.codes))
            return

        self.codes = reference_df["code"].astype(str).str.strip().tolist()
        self.descriptions = reference_df["description"].astype(str).str.strip().tolist()
        model = self.load_model()
        embeddings = model.encode(
            self.descriptions,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.embeddings = embeddings.astype("float32")
        np.save(embeddings_path, self.embeddings)
        metadata_path.write_text(
            json.dumps({"codes": self.codes, "descriptions": self.descriptions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._build_faiss_or_numpy(faiss_path)
        logger.info("Built semantic index with %s OKPD-2 descriptions", len(self.codes))

    def search(self, queries: Sequence[str], top_k: int = 50) -> List[List[OKPDCandidate]]:
        if self.embeddings is None or self.index is None:
            raise ValueError("Semantic index is not initialized.")

        model = self.load_model()
        query_vectors = model.encode(
            list(queries),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        top_k = min(top_k, len(self.codes))
        scores, indices = self.index.search(query_vectors, top_k)
        results: List[List[OKPDCandidate]] = []
        for row_scores, row_indices in zip(scores, indices):
            row = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue
                row.append(
                    OKPDCandidate(
                        code=self.codes[int(idx)],
                        description=self.descriptions[int(idx)],
                        retrieval_score=float(score),
                    )
                )
            results.append(row)
        return results

    def _try_import_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            self.faiss = faiss
        except ImportError:
            self.faiss = None
            logger.warning("faiss-cpu is not installed; using numpy inner-product search fallback.")

    def _build_faiss_or_numpy(self, faiss_path: Path) -> None:
        if self.faiss is None:
            self._build_numpy_index()
            return

        assert self.embeddings is not None
        index = self.faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        self.faiss.write_index(index, str(faiss_path))
        self.index = index

    def _build_numpy_index(self) -> None:
        assert self.embeddings is not None
        self.index = _NumpyIndex(self.embeddings)


class _NumpyIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings

    def search(self, query_vectors: np.ndarray, top_k: int):
        scores = query_vectors @ self.embeddings.T
        top_indices = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores.astype("float32"), top_indices.astype("int64")

