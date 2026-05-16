"""Построение трёхуровневого FAISS-индекса для иерархического поиска ОКПД-2."""
import pandas as pd
import numpy as np
from pathlib import Path
from config import config
from sentence_transformers import SentenceTransformer

def build_indexes():
    # Загружаем классификатор
    ref = pd.read_excel(config.OKPD2_REFERENCE_FILE)
    # Приводим колонки к единому виду
    code_col = next((c for c in ref.columns if str(c).strip().lower() in ('code', 'okpd', 'okpd2')), None)
    desc_col = next((c for c in ref.columns if str(c).strip().lower() in ('description', 'desc', 'name', 'title')), None)
    parent_col = next((c for c in ref.columns if str(c).strip().lower() in ('parent_code', 'parent')), None)
    if code_col is None or desc_col is None:
        raise ValueError("Не найдены колонки code/description в классификаторе")

    # Извлекаем коды и описания
    codes = ref[code_col].astype(str).str.strip().tolist()
    descriptions = ref[desc_col].astype(str).str.strip().tolist()

    # Определяем уровни иерархии
    class_level = {}   # ключ: первые 2 цифры, значение: список описаний
    group_level = {}   # ключ: первые 5 цифр, значение: список описаний
    code_level  = {}   # ключ: полный код, значение: описание

    for code, desc in zip(codes, descriptions):
        if len(code) < 2:
            continue
        cl = code[:2]
        gr = code[:5] if len(code) >= 5 else cl
        # Для класса берём описание самого класса (родительский код cl)
        class_level.setdefault(cl, desc)
        # Для группы – если есть родительская группа gr, берём её описание, иначе desc
        group_level.setdefault(gr, desc)
        # Полный код
        code_level[code] = desc

    # Загружаем Bi-Encoder (тот же, что в семантическом пайплайне)
    model = SentenceTransformer(config.SEMANTIC_BI_ENCODER_MODEL)
    base_dir = config.FAISS_DIR

    # Строим индексы и сохраняем
    import faiss

    for level_name, data in [("class", class_level), ("group", group_level), ("code", code_level)]:
        items = list(data.items())
        if not items:
            continue
        keys, texts = zip(*items)
        embeddings = model.encode(list(texts), show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # inner product для косинусного сходства
        faiss.normalize_L2(embeddings)        # нормализуем векторы
        index.add(embeddings)

        # Сохраняем индекс и список ключей
        save_dir = base_dir / f"{level_name}_level"
        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_dir / "index.faiss"))
        with open(save_dir / "keys.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(keys))
        print(f"✅ Уровень {level_name}: сохранено {len(keys)} записей")

if __name__ == "__main__":
    build_indexes()