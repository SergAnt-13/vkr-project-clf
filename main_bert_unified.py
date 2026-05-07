#!/usr/bin/env python3
"""
Точка входа для BERT пайплайна.
"""

from cli import main


def choose_mode() -> str:
    print("BERT классификатор ОКПД2")
    print("1. enhanced  - использовать merged-файл и эталонные метки, если они есть")
    print("2. standard  - использовать текущие коды из основной номенклатуры")

    while True:
        choice = input("Введите номер режима (1-2): ").strip()
        if choice == "1":
            return "enhanced"
        if choice == "2":
            return "standard"
        print("Введите 1 или 2.")


if __name__ == "__main__":
    mode = choose_mode()
    raise SystemExit(main(["bert", "--mode", mode]))
