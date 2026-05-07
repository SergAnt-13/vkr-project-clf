# BERT Notes

Этот файл оставлен как краткая историческая заметка.

Актуальная документация по запуску проекта теперь находится в:

- `README.md`
- `docs/WORKFLOW.md`

Текущая точка входа для BERT:

```bash
./venv/bin/python cli.py bert --mode enhanced
```

Если нужно использовать отдельный обучающий файл:

```bash
./venv/bin/python cli.py bert \
  --mode standard \
  --training-file "/путь/к/файлу.xlsx"
```
