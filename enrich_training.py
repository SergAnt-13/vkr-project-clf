"""
Расширение обучающей выборки за счёт подтверждённых кодов ОКПД2
из большого файла (52К), где уже были проставлены коды.
"""
import pandas as pd
from config import config

def find_column(columns, candidates):
    """Найти первую подходящую колонку из списка candidates."""
    for col in candidates:
        if col in columns:
            return col
    return None

def main():
    # === 1. Загружаем проверенные 1475 записей ===
    merged_path = config.MERGED_PRODUCTS_FILE
    labeled = pd.read_excel(merged_path)

    # Ищем колонку с названием товара
    name_col = find_column(labeled.columns, ['Номенклатура', 'номенклатура', 'name_raw', 'наименование', 'name'])
    # Ищем колонку с кодом ОКПД2
    okpd_col = find_column(labeled.columns, ['Код ОКПД2', 'код окпд2', 'okpd2_current', 'okpd2', 'окпд2'])

    if name_col is None or okpd_col is None:
        print("Не удалось найти нужные колонки в merged-файле. Доступные колонки:")
        print(labeled.columns.tolist())
        return

    labeled = labeled[[name_col, okpd_col]].dropna()
    labeled.columns = ['name_raw', 'okpd2_current']
    print(f"Проверенных записей (из merged): {len(labeled)}")

    # === 2. Загружаем большой файл с исходными кодами ===
    source_path = config.TO_PROCESS_DIR / "Номенклатурные единицы.xlsx"
    source = pd.read_excel(source_path)
    source_okpd_col = find_column(source.columns, ['Код ОКПД2', 'код окпд2', 'okpd2_current', 'okpd2'])
    source_name_col = find_column(source.columns, ['Номенклатура', 'номенклатура', 'name_raw', 'наименование', 'name'])
    if source_okpd_col is None or source_name_col is None:
        print("Не удалось найти нужные колонки в большом файле.")
        return
    # Оставляем только строки с заполненным кодом ОКПД2
    source = source[source[source_okpd_col].notna()].copy()
    # Переименовываем для удобства
    source.rename(columns={source_name_col: 'name_raw', source_okpd_col: 'original_okpd'}, inplace=True)
    print(f"Записей с исходными кодами в большом файле: {len(source)}")

    # === 3. Загружаем предсказания BERT ===
    pred_path = config.BERT_OUTPUT_DIR / "Номенклатурные единицы_bert_enhanced.xlsx"
    preds = pd.read_excel(pred_path)
    # Переименовываем колонки, которые нам нужны
    preds.rename(columns={
        'Идентификатор номенклатуры': 'item_id',
        'Код ОКПД2 (предсказанный)': 'okpd2_pred',
        'Уверенность предсказания': 'conf'
    }, inplace=True)
    print(f"Записей с предсказаниями: {len(preds)}")

    # === 4. Объединяем по item_id ===
    # Убедимся, что в source есть item_id (Идентификатор номенклатуры)
    if 'item_id' not in source.columns:
        source.rename(columns={'Идентификатор номенклатуры': 'item_id'}, inplace=True)
    merged = source.merge(
        preds[['item_id', 'okpd2_pred', 'conf']],
        on='item_id',
        how='inner'
    )
    print(f"Объединённых записей: {len(merged)}")

    # === 5. Отбираем высокоуверенные предсказания (с учётом бустинга) ===
    # Берём все записи, где уверенность > 0.8, не требуя совпадения с исходным кодом.
    # Это включает как старые подтверждённые, так и исправленные бустингом.
    high_conf = merged[(merged['conf'] > 0.8) & merged['okpd2_pred'].notna()].copy()
    print(f"Высокоуверенных предсказаний (conf > 0.8): {len(high_conf)}")

    # === 6. Добавляем их к проверенным ===
    high_conf = high_conf[['name_raw', 'okpd2_pred']].rename(columns={'okpd2_pred': 'okpd2_current'})
    enriched = pd.concat([labeled, high_conf], ignore_index=True)
    enriched = enriched.drop_duplicates(subset=['name_raw'])
    print(f"Итоговый размер расширенной обучающей выборки: {len(enriched)}")

    # === 7. Сохраняем ===
    output_path = config.TRAINING_DIR / "Номенклатурные единицы_enriched.xlsx"
    enriched.to_excel(output_path, index=False)
    print(f"Файл сохранён: {output_path}")

    # === 8. Сохраняем расхождения для эксперта ===
    discrepancies = merged[
        merged['original_okpd'].astype(str).str.strip() != merged['okpd2_pred'].astype(str).str.strip()
    ]
    disc_path = config.OUTPUT_DIR / "for_expert_review_discrepancies.xlsx"
    discrepancies.to_excel(disc_path, index=False)
    print(f"Расхождений (коды не совпали): {len(discrepancies)}. Сохранено в {disc_path}")

if __name__ == "__main__":
    main()