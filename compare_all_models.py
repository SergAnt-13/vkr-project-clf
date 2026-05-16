import pandas as pd
import re
from collections import Counter
from pathlib import Path
from config import config

def analyze_file(file_path: Path, label: str) -> dict:
    """Извлечь статистику из одного Excel‑файла."""
    if not file_path.exists():
        print(f"⚠️  {label}: файл не найден ({file_path})")
        return {}

    df = pd.read_excel(file_path)

    # Определяем колонки (для BERT и semantic они разные)
    if "Уверенность предсказания" in df.columns:
        conf_col = "Уверенность предсказания"
    elif "score" in df.columns:
        conf_col = "score"
    elif "conf" in df.columns:
        conf_col = "conf"
    else:
        print(f"❌ {label}: не найдена колонка с уверенностью")
        return {}

    name_col = "Номенклатура" if "Номенклатура" in df.columns else "name_raw"

    total = len(df)
    high = (df[conf_col] > 0.7).sum()
    mid = ((df[conf_col] >= 0.3) & (df[conf_col] <= 0.7)).sum()
    low = (df[conf_col] < 0.3).sum()

    # Топ‑20 проблемных слов (уверенность < 0.5, взвешенные по сомнению)
    low_conf = df[df[conf_col] < 0.5].copy()
    low_conf["weight"] = 1.0 - low_conf[conf_col]
    word_weights = Counter()
    for _, row in low_conf.iterrows():
        name = str(row[name_col]).lower()
        clean = re.sub(r"[^а-яa-z0-9\s]", " ", name)
        tokens = clean.split()
        w = row["weight"]
        for token in tokens:
            word_weights[token] += w

    print(f"\n{'='*50}")
    print(f"📊 {label}")
    print(f"Всего записей: {total}")
    print(f"🟢 Уверенные (>0.7): {high} ({high/total*100:.1f}%)")
    print(f"🟡 Сомнительные (0.3-0.7): {mid} ({mid/total*100:.1f}%)")
    print(f"🔴 Неуверенные (<0.3): {low} ({low/total*100:.1f}%)")
    print(f"Средняя уверенность: {df[conf_col].mean():.4f}")
    print(f"Медианная уверенность: {df[conf_col].median():.4f}")
    print(f"\nТоп‑20 проблемных слов:")
    for word, weight in word_weights.most_common(50):
        print(f"  {word}: {weight:.1f}")

    return {
        "total": total,
        "high": high,
        "mid": mid,
        "low": low,
        "mean_conf": df[conf_col].mean(),
        "median_conf": df[conf_col].median(),
        "top_problem_words": word_weights.most_common(50),
    }

# === ТОЧКИ ВХОДА ===
print("🔍 Сравнительный анализ BERT, BERT Enriched и Semantic")

# 1. Обычный BERT (merged)
bert_merged_file = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_merged" / "Номенклатурные единицы_bert_enhanced.xlsx"
analyze_file(bert_merged_file, "BERT (merged)")

# 2. BERT Enriched
bert_enriched_file = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_enriched" / "Номенклатурные единицы_bert_enhanced.xlsx"
analyze_file(bert_enriched_file, "BERT (Enriched)")

# 3. Semantic (самый свежий)
semantic_dir = config.SEMANTIC_OUTPUT_DIR
semantic_files = list(semantic_dir.glob("*/Номенклатурные единицы_semantic.xlsx"))
if semantic_files:
    semantic_file = max(semantic_files, key=lambda p: p.parent.stat().st_mtime)
    analyze_file(semantic_file, f"Semantic ({semantic_file.parent.name})")
else:
    # fallback – ищем старый формат
    # 3. Semantic (явно указываем полный файл)
    semantic_file = config.SEMANTIC_OUTPUT_DIR / "20260516_014312" / "Номенклатурные единицы_semantic_20260516_022225.xlsx"
    if semantic_file.exists():
        analyze_file(semantic_file, "Semantic (52K)")
    else:
        print("❌ Файл не найден, проверьте путь")