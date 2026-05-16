"""
Обогащение merged-выборки гиперонимами из тезауруса.
1. Лемматизирует проблемные слова из problem_words.txt.
2. Запрашивает гиперонимы через ConceptNet (если тезаурус ещё не собран).
3. Добавляет синтетические примеры в merged-выборку.
"""
import pandas as pd
import requests
import time
from pathlib import Path
from pymystem3 import Mystem
from config import config

# Загружаем merged
merged = pd.read_excel(config.MERGED_PRODUCTS_FILE)
# Ищем колонку с названием
name_col = "Номенклатура" if "Номенклатура" in merged.columns else "name_raw"
okpd_col = "Код ОКПД2" if "Код ОКПД2" in merged.columns else "okpd2_current"

# Загружаем проблемные слова
with open("problem_words.txt", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]

# Лемматизация
mystem = Mystem()
lemmatized = {}
for word in words:
    lemma = mystem.lemmatize(word)[0].strip()
    if lemma and lemma != word:
        lemmatized[word] = lemma

# Загружаем или строим тезаурус
thesaurus_path = Path("food_thesaurus.xlsx")
thesaurus = {}
if thesaurus_path.exists():
    thesaurus_df = pd.read_excel(thesaurus_path)
    for _, row in thesaurus_df.iterrows():
        term = str(row['term']).strip().lower()
        hyper = str(row['hypernyms']).strip()
        if term and hyper:
            thesaurus[term] = hyper.split("; ")
else:
    # Строим через ConceptNet
    for word in words[:50]:  # ограничим 50 словами для скорости
        candidates = [word]
        if word in lemmatized:
            candidates.append(lemmatized[word])
        hypernyms = []
        for candidate in candidates:
            url = f"http://api.conceptnet.io/query?node=/c/ru/{candidate}&rel=/r/IsA&limit=3"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    for edge in resp.json().get("edges", []):
                        if edge["start"]["label"].lower() == candidate:
                            hypernyms.append(edge["end"]["label"])
            except Exception:
                pass
            time.sleep(0.3)
        if hypernyms:
            thesaurus[word] = list(dict.fromkeys(hypernyms))[:3]
            print(f"{word} -> {thesaurus[word]}")

# Создаём синтетические примеры
synthetic = []
for word, hypers in thesaurus.items():
    # Ищем код, к которому относится это слово в merged
    # (упрощённо: берём первый попавшийся код для товаров, содержащих это слово)
    matching = merged[merged[name_col].astype(str).str.contains(word, case=False, na=False)]
    if not matching.empty:
        code = matching.iloc[0][okpd_col]
        for hyper in hypers[:2]:
            synthetic.append({name_col: f"{word} {hyper}", okpd_col: code})

if synthetic:
    synthetic_df = pd.DataFrame(synthetic)
    merged = pd.concat([merged, synthetic_df], ignore_index=True)
    output_path = config.TRAINING_DIR / "Номенклатурные единицы_merged_enriched.xlsx"
    merged.to_excel(output_path, index=False)
    print(f"✅ Создано {len(synthetic)} синтетических примеров, сохранено в {output_path}")
else:
    print("❌ Не удалось создать синтетические примеры")