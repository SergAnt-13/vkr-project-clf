"""Анализ уверенности модели по кодам ОКПД2 (средняя, медианная, доля уверенных)."""
import pandas as pd
import re
from collections import Counter
from pathlib import Path
from config import config

# Найди последний Enriched-файл
enriched_dir = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_enriched"
files = list(enriched_dir.glob("**/Номенклатурные единицы_bert_enhanced*.xlsx"))
if not files:
    raise FileNotFoundError("Не найден выходной файл BERT Enriched")
latest_file = max(files, key=lambda p: p.stat().st_mtime)
print(f"Анализирую: {latest_file}")

df = pd.read_excel(latest_file)

# Колонки
conf_col = "Уверенность предсказания" if "Уверенность предсказания" in df.columns else "conf"
code_col = "Код ОКПД2 (предсказанный)" if "Код ОКПД2 (предсказанный)" in df.columns else "okpd2_pred"

# Группируем по кодам
class_stats = df.groupby(code_col)[conf_col].agg(["mean", "median", "count"]).reset_index()
class_stats.columns = ["code", "avg_conf", "median_conf", "count"]
class_stats["high_conf_share"] = df.groupby(code_col)[conf_col].apply(
    lambda x: (x > 0.7).mean()
).values

# Сортируем по средней уверенности (от меньшей к большей)
class_stats = class_stats.sort_values("avg_conf")

print("\n=== Топ-20 кодов с НАИМЕНЬШЕЙ средней уверенностью ===")
for _, row in class_stats.head(20).iterrows():
    print(f"  {row['code']}: ср.={row['avg_conf']:.4f}, мед.={row['median_conf']:.4f}, "
          f"записей={int(row['count'])}, увер.>0.7={row['high_conf_share']:.1%}")

print("\n=== Топ-20 кодов с НАИБОЛЬШЕЙ средней уверенностью ===")
for _, row in class_stats.tail(20).iloc[::-1].iterrows():
    print(f"  {row['code']}: ср.={row['avg_conf']:.4f}, мед.={row['median_conf']:.4f}, "
          f"записей={int(row['count'])}, увер.>0.7={row['high_conf_share']:.1%}")