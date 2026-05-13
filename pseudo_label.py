import pandas as pd
from pathlib import Path
from config import config

def main():
    # Файлы
    merged_path = config.MERGED_PRODUCTS_FILE
    pred_path = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_merged_bert_enhanced.xlsx"
    output_path = config.TRAINING_DIR / "Номенклатурные единицы_extended.xlsx"

    # 1. Загружаем исходные 1475 размеченных записей
    print("Читаем размеченные данные...")
    labeled_df = pd.read_excel(merged_path)
    labeled_df = labeled_df.rename(columns={'Номенклатура': 'name_raw', 'Код ОКПД2': 'okpd2_current'})
    labeled_df = labeled_df[['name_raw', 'okpd2_current']].dropna(subset=['okpd2_current'])
    print(f"Исходных размеченных: {len(labeled_df)}")

    # 2. Загружаем предсказания модели на 50К товаров
    print("Читаем предсказания...")
    pred_df = pd.read_excel(pred_path)
    high_conf = pred_df[(pred_df['conf'] >= 0.95) & pred_df['okpd2_pred'].notna()].copy()
    high_conf = high_conf.rename(columns={'okpd2_pred': 'okpd2_current'})
    high_conf = high_conf[['name_raw', 'okpd2_current']]
    print(f"Уверенных предсказаний: {len(high_conf)}")

    # 3. Объединяем и сохраняем
    extended_df = pd.concat([labeled_df, high_conf], ignore_index=True)
    extended_df.to_excel(output_path, index=False)
    print(f"Расширенный датасет ({len(extended_df)} записей) сохранён в {output_path}")

if __name__ == "__main__":
    main()