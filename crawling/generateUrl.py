# 정제된 경기 일정을 바탕으로 각 경기별로 크롤링할 고유 URL을 생성 및 저장합니다.
# 각 팀, 날짜별로 KBO 및 네이버 스포츠 URL 목록 csv를 생성합니다.


import pandas as pd
import re
from google.colab import files

# 팀 약어 고정값으로 매핑
team_abbr = {
    "한화": "HH", "KIA": "HT", "KT": "KT", "LG": "LG", "NC": "NC",
    "두산": "OB", "롯데": "LT", "삼성": "SS", "SSG": "SK", "키움": "WO"
}


def generate_kbo_game_urls(year):
    result = []
    gyear = year
    input_path = f"/content/KBO_{year}_schedule.csv"
    output_path = f"/content/KBO_{year}_URL.csv"

    try:
        df = pd.read_csv(input_path, header=None)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다 : {input_path}")
        return

    for _, row in df.iterrows():
        month = str(row[0]).strip()
        date_raw = str(row[1]).strip()
        match_cell = str(row[3]).strip() if len(row) > 3 else ""

        if not re.match(r"\d{2}\.\d{2}", date_raw):
            continue

        # 날짜 정제
        date_digits = re.sub(r"\D", "", date_raw)
        current_date = f"{gyear}{date_digits}"

        if "vs" in match_cell:
            try:
                away_raw, home_raw = re.split("vs", match_cell)
                away_team = re.sub(r"\d+", "", away_raw).strip()
                home_team = re.sub(r"\d+", "", home_raw).strip()

                if home_team not in team_abbr or away_team not in team_abbr:
                    print(f"팀 약어가 존재하지 않습니다 : {away_team}, {home_team}")
                    continue

                team_code = f"{current_date}{team_abbr[away_team]}{team_abbr[home_team]}0"
                kbo_url = f"https://www.koreabaseball.com/Game/LiveText.aspx?leagueId=1&seriesId=0&gameId={team_code}&gyear={gyear}"
                naver_url = f"https://m.sports.naver.com/game/{team_code}{gyear}/lineup"

                result.append([current_date, home_team, away_team, team_code, gyear, kbo_url, naver_url])
            except Exception as e:
                print(f"파싱 실패 : {match_cell} → {e}")

    if not result:
        print(f"URL이 생성되지 않았습니다: {year}")
        return

    df_result = pd.DataFrame(result, columns=[
        "날짜", "홈팀", "원정팀", "팀코드", "연도", "KBO문자URL", "네이버라인업URL"
    ])
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{year}년 URL 생성 완료 → {output_path}")
    files.download(output_path)

# 연도별 반복 실행
for year in [2023, 2024, 2025]:
    generate_kbo_game_urls(year)