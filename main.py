#!/usr/bin/env python3
"""
Обратносуместимая точка входа для базового пайплайна.
"""

from cli import main


if __name__ == "__main__":
    raise SystemExit(main(["baseline"]))
