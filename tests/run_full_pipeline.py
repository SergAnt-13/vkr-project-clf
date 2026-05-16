"""
Единый production-пайплайн:
1. Подготовка тезауруса (если нужно).
2. Обучение BERT на merged_enriched.
3. Предсказание на 52К с иерархическим семантическим вторым мнением.
"""
import subprocess
import sys
from pathlib import Path
from config import config

def run(cmd, description):
    print(f"\n>>> {description}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Ошибка на шаге: {description}")
        sys.exit(result.returncode)

def main():
    # Шаг 0: проверяем, есть ли уже обогащённый merged
    merged_enriched = config.TRAINING_DIR / "Номенклатурные единицы_merged_enriched.xlsx"
    if not merged_enriched.exists():
        print("⚠️  Обогащённый merged не найден, запускаем подготовку тезауруса...")
        # Собираем проблемные слова (если ещё нет)
        if not Path("problem_words.txt").exists():
            run(".\.venv\Scripts\python collect_problem_words.py", "Сбор проблемных слов")
        # Строим тезаурус (если ещё нет)
        if not Path("food_thesaurus.xlsx").exists():
            run(".\.venv\Scripts\python build_thesaurus.py", "Построение тезауруса")
        # Обогащаем merged
        run(".\.venv\Scripts\python enrich_with_thesaurus.py", "Обогащение merged-выборки")
        print("✅ Обогащённый merged создан.")

    # Шаг 1: обучение BERT на обогащённом merged (если модель ещё не сохранена)
    model_dir = config.MODEL_DIR
    if not (model_dir / "pytorch_model.bin").exists():
        run(
            f".\.venv\Scripts\python cli.py bert --mode enhanced "
            f'--training-file "{merged_enriched}" --epochs 13',
            "Обучение BERT на обогащённой выборке"
        )
    else:
        print("ℹ️  Модель уже обучена, пропускаем обучение.")

    # Шаг 2: предсказание на 52К с вторым мнением
    run(
        ".\.venv\Scripts\python cli.py bert --mode enhanced "
        f'--training-file "{merged_enriched}" --load-existing-model',
        "Предсказание на 52К с иерархическим семантиком"
    )

    print("\n🎉 Полный пайплайн завершён. Результаты в data/results/bert/")

if __name__ == "__main__":
    main()