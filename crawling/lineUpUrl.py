# KBO/네이버 URL로부터 Selenium을 사용하여 경기별 라인업 텍스트 정보(HTML)를 수집합니다.
# 파일로 각 경기의 라인업 텍스트를 저장합니다.


# setup 실행
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import time
import pandas as pd
from tqdm import tqdm
from google.colab import files
import os

# 단일 연도 처리 함수
def crawl_lineups_by_year(year, limit=10):
    url_csv_path = f"/content/KBO_{year}_URL.csv"

    try:
        df = pd.read_csv(url_csv_path)
    except FileNotFoundError:
        print(f"URL 파일을 찾을 수 없습니다 : {url_csv_path}")
        return

    urls = df["네이버라인업URL"].dropna().tolist()[:limit]
    all_data = []

    def get_lineup_text(url):
        try:
            print(f"접속 중 : {url}")
            driver.get(url)
            time.sleep(5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            tab_div = soup.find("div", class_="LineupTab_comp_lineup_tab__30R4L")

            if tab_div is None:
                print("라인업 정보를 찾을 수 없습니다.")
                return None

            return tab_div.get_text(separator="\n", strip=True)

        except Exception as e:
            print(f"오류 발생 : {e}")
            return None

    for url in tqdm(urls):
        try:
            game_id = url.split("/game/")[1].split("/")[0]  # 예: 20250709HTHH02025
        except IndexError:
            print(f"URL 파싱 실패: {url}")
            continue

        lineup_text = get_lineup_text(url)

        if lineup_text:
            save_path = f"/content/lineup_{game_id}.txt"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(lineup_text)
            all_data.append((game_id, url, save_path))

    # 결과 저장
    if all_data:
        result_path = f"/content/KBO_{year}_lineup_text.csv"
        df_result = pd.DataFrame(all_data, columns=["게임ID", "네이버라인업URL", "파일경로"])
        df_result.to_csv(result_path, index=False, encoding="utf-8-sig")
        print(f"{year}년 라인업 수집 완료 → {result_path}")
        files.download(result_path)
    else:
        print(f"라인업 데이터가 수집되지 않았습니다: {year}")

# 실행
for year in [2023, 2024, 2025]:
    crawl_lineups_by_year(year, limit=10) 
    # limit 값을 조정해 더 많은 경기 수집 가능