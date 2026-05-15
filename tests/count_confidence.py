"""
Подсчет количества записей в интервалах уверенности
по результатам production-прогона BERT (enriched).
"""
import pandas as pd
from config import config

def main():
    # Путь к файлу предсказаний финального enriched-прогона
    pred_path = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_enriched" / "Номенклатурные единицы_bert_enhanced.xlsx"

    df = pd.read_excel(pred_path)
    conf = df['Уверенность предсказания']

    total = len(conf)
    high = (conf > 0.7).sum()
    mid = ((conf >= 0.3) & (conf <= 0.7)).sum()
    low = (conf < 0.3).sum()

    print(f"Всего записей: {total}")
    print(f"🟢 Высокая уверенность (>0.7): {high} ({high/total*100:.1f}%)")
    print(f"🟡 Средняя (0.3–0.7): {mid} ({mid/total*100:.1f}%)")
    print(f"🔴 Низкая (<0.3): {low} ({low/total*100:.1f}%)")

if __name__ == "__main__":
    main()