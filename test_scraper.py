# src/scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"


def scrape_zakupki(query: str, num_pages: int = 1, save_to_file: bool = False,
                   output_filename: str = 'zakupki_data.xlsx') -> pd.DataFrame:
    """
    Парсинг сайта zakupki.gov.ru по ключевому запросу для получения данных по закупкам и сохранения в файл.

    :param query: GПоисковый запрос (например, 'IP камера')
    :param num_pages: количество страниц для парсинга
    :param save_to_file: флаг для сохранения результатов в файл
    :param output_filename: имя файла для сохранения результатов
    :return: DataFrame с наименованиями закупок и кодами ОКПД2
    """
    results = []

    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}?searchString={query}&pageNumber={page}"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            items = soup.find_all('div', class_='search-registry-entry-block')
            for item in items:
                try:
                    name = item.find('div', class_='registry-entry__body-value').get_text(strip=True)
                    okpd2_code_element = item.find('span', class_='registry-entry__code')
                    if okpd2_code_element:
                        okpd2_code = okpd2_code_element.get_text(strip=True)
                    else:
                        okpd2_code = "Не найден"

                    results.append({
                        'Наименование': name,
                        'ОКПД2': okpd2_code
                    })
                except Exception as e:
                    print(f"Ошибка при парсинге элемента: {e}")
        else:
            print(f"Ошибка при запросе {url}, код ответа: {response.status_code}")

    df = pd.DataFrame(results)

    if save_to_file:
        df.to_excel(output_filename, index=False)
        print(f"Данные успешно сохранены в файл {output_filename}")

    return df