#!/usr/bin/env python3
"""
Интерактивный запуск проекта.
"""

from cli import main


def choose_command() -> list:
    print("Классификатор ОКПД2")
    print("1. inspect-data  - посмотреть, какие данные видит проект")
    print("2. baseline      - запустить базовый TF-IDF + kNN пайплайн")
    print("3. bert          - запустить BERT пайплайн")

    while True:
        choice = input("Введите номер (1-3): ").strip()
        if choice == "1":
            return ["inspect-data"]
        if choice == "2":
            return ["baseline"]
        if choice == "3":
            mode = input("Режим BERT (enhanced/standard, по умолчанию enhanced): ").strip() or "enhanced"
            if mode not in {"enhanced", "standard"}:
                print("Режим должен быть enhanced или standard.")
                continue
            return ["bert", "--mode", mode]
        print("Введите 1, 2 или 3.")


if __name__ == "__main__":
    raise SystemExit(main(choose_command()))
