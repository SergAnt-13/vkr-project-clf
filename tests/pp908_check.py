import sys
from pathlib import Path

# Убедимся, что можем импортировать конфиг проекта
try:
    from config import Config
except ImportError:
    print("Не удалось импортировать Config. Проверьте, что вы запускаете скрипт из корня проекта.")
    sys.exit(1)

cfg = Config()
pp_path = cfg.PP908_FILE

print(f"Ожидаемый путь к файлу ПП-908: {pp_path.resolve()}")
print(f"Файл существует? {pp_path.exists()}")
print(f"Файл является файлом? {pp_path.is_file()}")
print(f"Размер файла: {pp_path.stat().st_size if pp_path.exists() else 'N/A'} байт")

if pp_path.exists():
    # Попробуем прочитать как обычный текст (UTF-8, игнорируя ошибки)
    try:
        content = pp_path.read_text(encoding="utf-8", errors="ignore")
        print(f"При чтении как UTF-8 получено {len(content)} символов. Первые 200 символов:")
        print(repr(content[:200]))
    except Exception as e:
        print(f"Ошибка при чтении как UTF-8: {e}")

    # Попробуем прочитать бинарно и вывести первые байты
    try:
        with open(pp_path, "rb") as f:
            raw = f.read(200)
        print(f"Первые 200 байт (в виде шестнадцатеричной строки): {raw.hex()}")
        print(f"Первые 200 байт (попытка декодировать как ascii, ignore): {raw.decode('ascii', errors='ignore')}")
    except Exception as e:
        print(f"Ошибка при бинарном чтении: {e}")

    # Проверим возможность конвертации через striprtf (если установлена)
    try:
        from striprtf.striprtf import rtf_to_text
        with open(pp_path, "r", encoding="utf-8", errors="ignore") as f:
            rtf_content = f.read()
        text = rtf_to_text(rtf_content)
        print(f"Конвертация через striprtf успешна: получено {len(text)} символов.")
        print("Первые 300 символов текста:")
        print(text[:300])
    except ImportError:
        print("Библиотека striprtf не установлена. Пропускаем проверку конвертации.")
    except Exception as e:
        print(f"Ошибка при конвертации через striprtf: {e}")

else:
    print("Файл не найден. Проверьте имя и расположение.")
