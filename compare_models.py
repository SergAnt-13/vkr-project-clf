"""
Сравнение метрик Baseline, BERT (merged) и BERT (enriched) в одной таблице.
"""
import pandas as pd
from pathlib import Path
from config import config

def extract_metrics_from_report(report_path: Path) -> dict:
    """Извлечь метрики из текстового отчёта."""
    metrics = {}
    if not report_path.exists():
        return metrics
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Accuracy' in line:
                try:
                    metrics['accuracy'] = float(line.split(':')[-1].strip())
                except:
                    pass
            if 'F1 (макро)' in line:
                try:
                    metrics['f1_macro'] = float(line.split(':')[-1].strip())
                except:
                    pass
            if 'F1 (взвешенная)' in line:
                try:
                    metrics['f1_weighted'] = float(line.split(':')[-1].strip())
                except:
                    pass
    return metrics

def main():
    results_dir = config.RESULTS_DIR
    rows = []

    # Baseline
    baseline_dir = results_dir / 'baseline'
    baseline_report = baseline_dir / 'summary_report.txt'
    baseline_metrics = extract_metrics_from_report(baseline_report)
    if baseline_metrics:
        baseline_metrics['model'] = 'Baseline (TF-IDF)'
        baseline_metrics['training_file'] = 'merged'
        rows.append(baseline_metrics)

    # BERT на merged
    bert_merged_dir = results_dir / 'bert' / 'Номенклатурные единицы_merged'
    bert_merged_report = bert_merged_dir / 'bert_enhanced_report.txt'
    bert_merged_metrics = extract_metrics_from_report(bert_merged_report)
    # Дополнительно можно достать иерархию, если она уже добавлена в отчёт
    if bert_merged_metrics:
        bert_merged_metrics['model'] = 'BERT (RuBERT)'
        bert_merged_metrics['training_file'] = 'merged'
        rows.append(bert_merged_metrics)

    # BERT на enriched
    bert_enriched_dir = results_dir / 'bert' / 'Номенклатурные единицы_enriched'
    bert_enriched_report = bert_enriched_dir / 'bert_enhanced_report.txt'
    bert_enriched_metrics = extract_metrics_from_report(bert_enriched_report)
    if bert_enriched_metrics:
        bert_enriched_metrics['model'] = 'BERT (Enriched)'
        bert_enriched_metrics['training_file'] = 'enriched'
        rows.append(bert_enriched_metrics)

    if not rows:
        print("Не найдено ни одного отчёта. Сначала запустите модели.")
        return

    df = pd.DataFrame(rows)
    # Переупорядочим колонки
    cols = ['model', 'training_file', 'accuracy', 'f1_macro', 'f1_weighted']
    df = df[[col for col in cols if col in df.columns]]

    print("\n=== СВОДКА МЕТРИК ===")
    print(df.to_string(index=False))

    out_path = results_dir / 'compare_report.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nСводка сохранена в {out_path}")

if __name__ == "__main__":
    main()