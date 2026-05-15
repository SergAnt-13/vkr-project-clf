"""
Единая CLI-точка входа для проекта.
"""

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

from config import config
from data_loader import DataLoader
from pipelines import BERTPipeline, BaselinePipeline
from semantic_pipeline import SemanticPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Классификация номенклатуры по ОКПД2 и расчет НДС.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="Показать, какие файлы проект видит и сколько в них размеченных данных.",
    )
    inspect_parser.set_defaults(func=command_inspect_data)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Запустить базовый TF-IDF + kNN пайплайн.",
    )
    add_common_dataset_args(baseline_parser)
    baseline_parser.set_defaults(func=command_baseline)

    bert_parser = subparsers.add_parser(
        "bert",
        help="Запустить BERT пайплайн.",
    )
    add_common_dataset_args(bert_parser)
    bert_parser.add_argument(
        "--mode",
        choices=["standard", "enhanced", "semantic"],
        default="enhanced",
        help="enhanced = пытаться учиться на эталонных метках из merged-файла.",
    )
    bert_parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.MODEL_DIR,
        help="Каталог для сохранения или загрузки модели.",
    )
    bert_parser.add_argument(
        "--load-existing-model",
        action="store_true",
        help="Не обучать заново, а загрузить уже сохраненную модель.",
    )
    bert_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N products in semantic mode.",
    )
    bert_parser.add_argument(
        "--skip-semantic-finetune",
        action="store_true",
        help="Skip optional bi-encoder fine-tuning in semantic mode.",
    )
    bert_parser.set_defaults(func=command_bert)

    return parser


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=None,
        help="Файл, для которого нужно получить предсказания.",
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=None,
        help="Отдельный файл для обучения. Если не указан, обучение идет на встроенных данных проекта.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Каталог для результатов.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Ограничить количество строк в файле предсказаний.",
    )
    parser.add_argument(
        "--training-max-rows",
        type=int,
        default=None,
        help="Ограничить количество строк в обучающем файле.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Обучить и оценить модель на отложенной части тренировочных данных (без предсказаний на основном файле).",
    )


def command_inspect_data(_: argparse.Namespace) -> int:
    loader = DataLoader(config)
    sources = [
        ("Основная номенклатура", config.PRODUCTS_FILE),
        ("Merged-файл", config.MERGED_PRODUCTS_FILE),
        ("Эталон НДС", config.VAT_REFERENCE_FILE),
        ("Сокращения", config.ABBREVIATIONS_FILE),
    ]

    for title, path in sources:
        print(f"\n=== {title} ===")
        if not path.exists():
            print(f"Файл не найден: {path}")
            continue

        if path.name == config.ABBREVIATIONS_FILE.name:
            rules = loader.load_abbreviations()
            print(f"Файл: {path}")
            print(f"Правил сокращений: {len(rules)}")
            continue

        info = loader.describe_source(path)
        print(f"Файл: {info['path']}")
        print(f"Строк: {info['rows']}")
        print(f"Колонки: {', '.join(info['columns'])}")
        print(f"Валидных ОКПД2 в текущей колонке: {info['valid_okpd_current']}")
        print(f"Уникальных текущих кодов: {info['unique_current_codes']}")
        if info["valid_okpd_reference"] > 0:
            print(f"Валидных эталонных ОКПД2: {info['valid_okpd_reference']}")
            print(f"Уникальных эталонных кодов: {info['unique_reference_codes']}")

    return 0


def command_baseline(args: argparse.Namespace) -> int:
    pipeline = BaselinePipeline(config)
    result = pipeline.run(
        prediction_file=args.prediction_file if not args.validate else None,
        training_file=args.training_file,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        training_max_rows=args.training_max_rows,
        validate=args.validate,
    )

    print("Базовый пайплайн завершен.")
    if not args.validate:
        print(f"Файл для предсказаний: {result['prediction_source']}")
        print(f"Файл для обучения: {result['training_source']}")
        for name, path in result["saved_files"].items():
            print(f"{name}: {path}")
    return 0


def command_bert(args: argparse.Namespace) -> int:
    if args.mode == "semantic":
        pipeline = SemanticPipeline(config)
        result = pipeline.run(
            prediction_file=args.prediction_file,
            training_file=args.training_file,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            training_max_rows=args.training_max_rows,
            limit=args.limit,
            fine_tune=not args.skip_semantic_finetune,
        )

        print("Semantic pipeline completed.")
        print(f"Prediction file: {result['prediction_source']}")
        print(f"Training file: {result['training_source']}")
        print(f"Result: {result['output_file']}")
        return 0

    pipeline = BERTPipeline(config)
    result = pipeline.run(
        mode=args.mode,
        prediction_file=args.prediction_file if not args.validate else None,
        training_file=args.training_file,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        load_existing_model=args.load_existing_model and not args.validate,
        max_rows=args.max_rows,
        training_max_rows=args.training_max_rows,
        validate=args.validate,
    )

    print("BERT пайплайн завершен.")
    if not args.validate:
        print(f"Файл для предсказаний: {result['prediction_source']}")
        print(f"Файл для обучения: {result['training_source']}")
        print(f"Каталог модели: {result['model_dir']}")
        print(f"Результат: {result['output_file']}")
        print(f"Отчет: {result['report_file']}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
