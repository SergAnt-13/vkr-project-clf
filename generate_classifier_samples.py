import pandas as pd
from config import config
from pathlib import Path

# Загружаем классификатор
ref = pd.read_excel(config.OKPD2_REFERENCE_FILE)
# Приводим колонки к единому виду
ref = ref.rename(columns={col: col.strip().lower() for col in ref.columns})

# Находим колонки
code_col = next((c for c in ref.columns if c in ('code', 'okpd', 'okpd2')), None)
desc_col = next((c for c in ref.columns if c in ('description', 'desc', 'name')), None)
parent_col = next((c for c in ref.columns if c in ('parent_code', 'parent')), None)

if code_col is None or desc_col is None:
    raise ValueError("Не найдены колонки code/description")

# Строим словарь родительских описаний (если есть parent_code)
parent_descriptions = {}
if parent_col is not None:
    for _, row in ref.iterrows():
        code = str(row[code_col]).strip()
        desc = str(row[desc_col]).strip()
        parent_descriptions[code] = desc

# Генерируем синтетические примеры
samples = []
for _, row in ref.iterrows():
    code = str(row[code_col]).strip()
    description = str(row[desc_col]).strip()

    # Строим иерархическую цепочку
    hierarchy_parts = []
    current = code
    while current in parent_descriptions and current != parent_descriptions.get(current, ''):
        parent_desc = parent_descriptions[current]
        if parent_desc:
            hierarchy_parts.insert(0, parent_desc)
        # Переходим к родительскому коду, если он есть
        parent_row = ref[ref[code_col] == current]
        if not parent_row.empty and parent_col is not None:
            current = str(parent_row.iloc[0].get(parent_col, '')).strip()
            if not current:
                break
        else:
            break

    # Собираем полный текст
    hierarchy_text = ". ".join(hierarchy_parts) + f". Код {code}: {description}" if hierarchy_parts else f"Код {code}: {description}"

    # Добавляем несколько вариантов для разнообразия
    # Вариант 1: полный текст
    samples.append({'name_raw': hierarchy_text, 'okpd2_current': code})
    # Вариант 2: только описание + код
    samples.append({'name_raw': f"{description} (код {code})", 'okpd2_current': code})

# Сохраняем
samples_df = pd.DataFrame(samples)
output_path = config.TRAINING_DIR / "classifier_training_samples.xlsx"
samples_df.to_excel(output_path, index=False)
print(f"✅ Сгенерировано {len(samples_df)} синтетических примеров, сохранено в {output_path}")