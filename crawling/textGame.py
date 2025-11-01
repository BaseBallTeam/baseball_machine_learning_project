# Selenium으로 게임별 상세 텍스트(이닝별 전개 등)을 크롤링 및 수집하여 텍스트 파일로 저장합니다.
# 저장된 라인업 csv 기반으로 대상 게임의 텍스트를 한 번에 수집합니다.


import pandas as pd
import re
import time
from tqdm import tqdm
from selenium.webdriver.common.by import By
from google.colab import files

# 중계 수집 함수 (이닝별로 [GAME_ID] 표시 포함)
def extract_multiple_games_compact(game_ids, output_path, year):
    all_text = ""

    for game_id in tqdm(game_ids, desc=f"📡 {year} 중계 수집"):
        url = f"https://www.koreabaseball.com/Game/LiveText.aspx?leagueId=1&seriesId=0&gameId={game_id}&gyear={year}"
        print(f"\n📡 중계 수집 중: {url}")
        game_text = f"[GAME_ID:{game_id}]\n"  # 경기 id는 1번만

        try:
            driver.get(url)
            time.sleep(6)

            for i in range(1, 11):
                div_id = f"numCont{i}"
                try:
                    driver.execute_script(f"document.getElementById('{div_id}').style.display = 'block';")
                    element = driver.find_element(By.ID, div_id)
                    inning_text = element.text.strip()
                    reversed_lines = "\n".join(inning_text.split("\n")[::-1])

                    game_text += f"\n[numCont{i}]\n{reversed_lines}\n"
                except:
                    print(f"{div_id} 불러오기 실패")
        except Exception as e:
            print(f"{game_id} 접속 오류 : {e}")
            continue

        all_text += game_text.strip() + "\n\n"

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text.strip())

    print(f"\n중계 텍스트 저장 완료: {output_path}")
    return output_path



for year in [2023, 2024, 2025]:
    csv_path = f"/content/KBO_{year}_lineup.csv"

    try:
        lineup_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"파일이 존재하지 않습니다: {csv_path}")
        continue

    lineup_df["게임ID"] = lineup_df["게임ID"].astype(str).str.strip()
    game_ids_full = lineup_df["게임ID"].dropna().unique().tolist()
    game_ids = sorted([gid[:13] for gid in game_ids_full if len(gid) >= 13])

    print(f"\n {year}년 경기 수 : {len(game_ids)}")
    print("추출된 게임 ID 예시 : ", game_ids[:3])

    output_path = f"/content/KBO_{year}_inning_text.txt"
    extract_multiple_games_compact(game_ids, output_path, year)
    files.download(output_path)
