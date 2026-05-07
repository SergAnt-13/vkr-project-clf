#!/usr/bin/env python3
"""
Legacy-совместимая точка входа для BERT пайплайна.
"""

from cli import main


if __name__ == "__main__":
    raise SystemExit(main(["bert", "--mode", "standard"]))
