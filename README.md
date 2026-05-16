# OKPD2 Classification Project

Проект предназначен для автоматической классификации номенклатурных позиций по кодам ОКПД2 и для расчета предполагаемой ставки НДС.

Это не веб-сервис и не библиотека, а прикладной конвейер обработки табличных выгрузок.

## Для чего проект нужен

На вход подается Excel-файл с номенклатурой товаров.

Проект:

1. читает таблицу и ищет в ней нужные колонки
2. нормализует названия товаров
3. выбирает строки, где уже есть известный код ОКПД2
4. использует эти строки как обучающую выборку
5. предсказывает код ОКПД2 для остальных строк
6. по итоговому коду пытается определить ставку НДС
7. сохраняет результат в отдельные выходные файлы

## Какие модели есть

- `baseline` — быстрый подход `TF-IDF + kNN`
- `bert` — более тяжелый подход на базе `DeepPavlov/rubert-base-cased`

## Что важно понимать про обучение

### Baseline

`baseline` не делает deep learning.

Он:

- очищает текст товара
- строит символьные TF-IDF признаки
- ищет наиболее похожие товары
- голосует по кодам ОКПД2 ближайших соседей

### BERT

`bert` использует русскую предобученную модель и дообучает ее на ваших размеченных названиях товаров и кодах ОКПД2.

## Текущая структура репозитория

- `cli.py` — единая командная точка входа
- `run.py` — интерактивный запуск
- `main.py` — совместимый запуск baseline
- `main_bert_unified.py` — совместимый запуск BERT с выбором режима
- `pipelines.py` — прикладные пайплайны baseline и BERT
- `data_loader.py`, `text_processor.py`, `okpd_classifier.py`, `bert_okpd_classifier.py`, `vat_processor.py`, `output_manager.py` — ядро обработки
- `data/` — входные данные и новые результаты
- `archive/` — старые артефакты и рудименты, не участвующие в текущем пайплайне
- `docs/` — человекоориентированная документация
- `tests/` — минимальные проверки

## Где теперь лежат данные

- `data/to_process/` — файлы, которые нужно обработать
- `data/training/` — обучающие файлы и размеченные таблицы
- `data/reference/` — справочники и нормативные файлы
- `data/results/` — результаты новых запусков

По умолчанию проект ожидает:

- `data/to_process/Номенклатурные единицы.xlsx`
- `data/training/Номенклатурные единицы_merged.xlsx`
- `data/reference/Товары ставка 10%.xlsx`
- `data/reference/сокращения.xlsx`
- `data/reference/Постановление Правительства РФ ... .rtf`

## Как формируется обучающая выборка сейчас

Проект работает с несколькими источниками:

- основная номенклатура: `data/to_process/Номенклатурные единицы.xlsx`
- merged-файл с эталонными метками: `data/training/Номенклатурные единицы_merged.xlsx`
- справочники и нормативные данные: `data/reference/...`

Логика обучения:

- `baseline` по умолчанию учится на размеченных строках prediction-файла
- `bert --mode standard` по умолчанию учится на текущих кодах основной номенклатуры
- `bert --mode enhanced` по умолчанию пытается учиться на эталонных метках merged-файла
- любой режим можно переключить на отдельный обучающий файл через `--training-file`

Это особенно важно для вашего сценария с новым почти корректным размеченным файлом на `1500-2000` строк.

## === Запуск и тест метрик ===

### 1 Проверить, что проект видит

```bash
./.venv/Scripts/python cli.py inspect-data
```

### 2 Тест Baseline для метрик

```bash
.\.venv\Scripts\python cli.py baseline --training-file "data/training/Номенклатурные единицы_merged.xlsx" --validate
```

### 3 Тест BERT для метрик (дообучение при наличии файла)

```bash
.\.venv\Scripts\python cli.py bert --mode enhanced --training-file "data/training/Номенклатурные единицы_merged.xlsx" --validate
```

## === Прогон моделей на реальных данных ===

### 1 Baseline на полной номенклатуре

```bash
.\.venv\Scripts\python cli.py baseline --training-file "data/training/Номенклатурные единицы_merged.xlsx"
```

### 2 BERT на исходном merged для обучения (1500 записей):

```bash
.\.venv\Scripts\python cli.py bert --mode enhanced --training-file "data/training/Номенклатурные единицы_merged.xlsx"
```

### 3 BERT на расширенном enriched для обучения (3500 записей)
```bash
.\.venv\Scripts\python enrich_training.py
```
```bash
.\.venv\Scripts\python cli.py bert --mode enhanced --training-file "data/training/Номенклатурные единицы_enriched.xlsx"
```

### 4 Использование последней выполненной модели для предсказания
```bash
.\.venv\Scripts\python cli.py bert --mode enhanced --load-existing-model
```

### 6 Semantic pipeline
```bash
.\.venv\Scripts\python cli.py bert --mode semantic --limit 20
```

### 6 Semantic pipeline без долгого дообучения
```bash
.\.venv\Scripts\python cli.py bert --mode semantic --skip-semantic-finetune
```




### 5 Сравнение моделей
```bash
.\.venv\Scripts\python compare_models.py
```

### 6 Интерактивный запуск

```bash
.\.venv\Scripts\python run.py
```

## Куда сохраняются результаты

- baseline: `data/results/baseline/`
- bert: `data/results/bert/`
- модели BERT: `artifacts/models/bert_okpd/`

## Какие документы читать

- [docs/WORKFLOW.md](/Users/sergant/Documents/Вытащил%20с%20диска/ML%20OKPD/okpd/docs/WORKFLOW.md:1) — пошаговый порядок действий
- [docs/DATA_FLOW.md](/Users/sergant/Documents/Вытащил%20с%20диска/ML%20OKPD/okpd/docs/DATA_FLOW.md:1) — что проект берет на вход, как обрабатывает и что отдает
- [docs/BUSINESS_CONTEXT.md](/Users/sergant/Documents/Вытащил%20с%20диска/ML%20OKPD/okpd/docs/BUSINESS_CONTEXT.md:1) — деловое описание системы без программистского жаргона
- [docs/GIT_PREP.md](/Users/sergant/Documents/Вытащил%20с%20диска/ML%20OKPD/okpd/docs/GIT_PREP.md:1) — что безопасно отправлять в Git, а что нельзя
- [docs/PROJECT_STRUCTURE.md](/Users/sergant/Documents/Вытащил%20с%20диска/ML%20OKPD/okpd/docs/PROJECT_STRUCTURE.md:1) — карта каталогов

## Что можно отправлять в Git

Обычно в Git должны идти:

- `.py` файлы с кодом
- `README.md`
- `docs/`
- `tests/`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`

Не должны идти:

- реальные Excel/CSV/RTF с бизнес-данными
- результаты модели
- локальные логи
- `venv/`
- `archive/`
- `artifacts/`
- `.idea/`

Для этого `.gitignore` уже настроен.

## Что уже вынесено из рабочего контура

В `archive/` перенесены:

- старые выходные Excel-файлы
- старые отчеты
- старые логи
- устаревшие/неиспользуемые данные
- локальные технические артефакты

То есть текущий рабочий контур и исторический хвост теперь разделены.
