import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config

model = SentenceTransformer(config.SEMANTIC_BI_ENCODER_MODEL)

def search_level(level: str, query: str, k: int = 5):
    index_path = config.FAISS_DIR / f"{level}_level" / "index.faiss"
    keys_path = config.FAISS_DIR / f"{level}_level" / "keys.txt"
    index = faiss.read_index(str(index_path))
    with open(keys_path, encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    vec = model.encode([query])
    faiss.normalize_L2(vec)
    dist, ids = index.search(vec, k)
    return [(keys[i], dist[0][j]) for j, i in enumerate(ids[0])]

# Примеры:
print("Классы, близкие к 'Молоко':", search_level("class", "Молоко", 5))
print("Группы, близкие к 'Молоко':", search_level("group", "Молоко", 5))
print("Коды, близкие к 'Шоколад':", search_level("code", "Шоколад", 5))