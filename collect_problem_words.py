"""
Собирает проблемные слова из двух источников:
1. Редкие слова из всей номенклатуры (частота <= 5).
2. Неуверенные слова из последнего предсказания Enriched BERT (уверенность < 0.5).
Объединяет их с весами и сохраняет в problem_words.txt для тезауруса.
"""
import pandas as pd
import re
from collections import Counter
from pathlib import Path
from config import config

# ------------------- 1. Редкие слова -------------------
df_all = pd.read_excel(config.PRODUCTS_FILE)
name_col = "Номенклатура" if "Номенклатура" in df_all.columns else "name_raw"
df_all[name_col] = df_all[name_col].fillna("").astype(str)
all_words = []
for name in df_all[name_col].astype(str).str.lower():
    clean = re.sub(r"[^а-яa-z\s]", " ", name)
    all_words.extend(clean.split())

word_counts = Counter(all_words)
rare_words = {w: c for w, c in word_counts.items() if c <= 5 and len(w) > 3}
print(f"Редких слов: {len(rare_words)}")

# ------------------- 2. Неуверенные слова -------------------
enriched_dir = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_enriched"
files = list(enriched_dir.glob("Номенклатурные единицы_bert_enhanced*.xlsx"))
if not files:
    raise FileNotFoundError("Не найден выходной файл BERT Enriched")
latest_file = max(files, key=lambda p: p.stat().st_mtime)
print(f"Анализирую: {latest_file}")

df_bert = pd.read_excel(latest_file)
conf_col = "Уверенность предсказания" if "Уверенность предсказания" in df_bert.columns else "conf"
name_col_bert = "Номенклатура" if "Номенклатура" in df_bert.columns else "name_raw"
df_bert[name_col_bert] = df_bert[name_col_bert].fillna("").astype(str)
low_conf = df_bert[df_bert[conf_col] < 0.5].copy()
low_conf["weight"] = 1.0 - low_conf[conf_col]

uncertain_weights = Counter()
MAX_WORDS = 3  # первые 3 слова названия
for _, row in low_conf.iterrows():
    name = str(row[name_col_bert]).lower()
    clean = re.sub(r"[^а-яa-z\s]", " ", name)
    tokens = clean.split()[:MAX_WORDS]
    w = row["weight"]
    for token in tokens:
        if len(token) > 2:
            uncertain_weights[token] += w

print(f"Неуверенных слов: {len(uncertain_weights)}")

# ------------------- 3. Объединяем -------------------
combined = Counter()

# Редкие слова получают базовый вес 1.0
for word in rare_words:
    combined[word] += 1.0

# Неуверенные слова — их накопленный вес (обычно > 1)
for word, weight in uncertain_weights.items():
    combined[word] += weight

# Берём топ-300 по суммарному весу
top_words = combined.most_common(300)

with open("problem_words.txt", "w", encoding="utf-8") as f:
    for word, _ in top_words:
        f.write(word + "\n")

print(f"✅ Сохранено {len(top_words)} проблемных слов в problem_words.txt")
