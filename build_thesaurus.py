import requests
import time
import pandas as pd
from pathlib import Path
from pymystem3 import Mystem

mystem = Mystem()

def lemmatize(word):
    """Привести слово к нормальной форме."""
    lemmas = mystem.lemmatize(word)
    # mystem возвращает список, берём первый элемент, отбрасываем пустые и пробелы
    for lemma in lemmas:
        lemma = lemma.strip()
        if lemma and lemma.isalpha():
            return lemma.lower()
    return word.lower()

def get_hypernyms(word):
    """Получить гиперонимы для слова через ConceptNet API."""
    # Сначала попробуем исходное слово, потом лемму
    candidates = [word]
    lemma = lemmatize(word)
    if lemma != word:
        candidates.append(lemma)

    hypernyms = []
    for candidate in candidates:
        url = f"http://api.conceptnet.io/query?node=/c/ru/{candidate}&rel=/r/IsA&limit=3"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for edge in data.get("edges", []):
                start = edge["start"]["label"].lower()
                if start == candidate:
                    hypernyms.append(edge["end"]["label"])
        except Exception:
            continue

    # Убираем дубликаты и возвращаем до 5 уникальных
    return list(dict.fromkeys(hypernyms))[:5]

def main():
    # Загружаем список редких слов (создан ранее через collect_problem_words.py)
    rare_file = Path("rare_words.txt")
    if not rare_file.exists():
        print("❌ Файл rare_words.txt не найден. Сначала запустите collect_problem_words.py")
        return

    with open("problem_words.txt", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    print(f"🔍 Найдено {len(words)} редких слов для обработки")

    thesaurus = {}
    for i, word in enumerate(words, 1):
        if i % 20 == 0:
            print(f"   Обработано {i}/{len(words)}...")
        hyper = get_hypernyms(word)
        if hyper:
            thesaurus[word] = hyper
            print(f"   {word} → {', '.join(hyper)}")
        time.sleep(0.3)  # вежливая пауза для API

    if thesaurus:
        df = pd.DataFrame(
            [{"term": w, "hypernyms": "; ".join(h)} for w, h in thesaurus.items()]
        )
        output_path = Path("food_thesaurus.xlsx")
        df.to_excel(output_path, index=False)
        print(f"✅ Тезаурус сохранён: {output_path} ({len(df)} терминов)")
    else:
        print("⚠️ Для данных слов гиперонимы не найдены. Попробуйте другие API или Wikidata.")

if __name__ == "__main__":
    main()