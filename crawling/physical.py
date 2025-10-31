# Selenium을 활용해 선수 등록 페이지에서 신체 정보(키, 몸무게 등)를 수집합니다.
# 연도 및 팀별로 전국 선수 신체 데이터 csv를 만듭니다.


import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
from google.colab import files

# CSV 파일 불러오기
df_2023 = pd.read_csv("/content/KBO_2023_URL.csv")
df_2024 = pd.read_csv("/content/KBO_2024_URL.csv")
df_2025 = pd.read_csv("/content/KBO_2025_URL.csv")

# 연도별 날짜 5개씩만 추출
dates_2023 = sorted(df_2023['날짜'].astype(str).unique())[:5]
dates_2024 = sorted(df_2024['날짜'].astype(str).unique())[:5]
dates_2025 = sorted(df_2025['날짜'].astype(str).unique())[:5]

# 합치기
valid_dates = dates_2023 + dates_2024 + dates_2025
print(f"📆 총 {len(valid_dates)}일 크롤링 예정:", valid_dates)

# 팀 리스트
teams = ['HH', 'LG', 'LT', 'HT', 'KT', 'SS', 'SK', 'NC', 'OB', 'WO']

# Chrome 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

all_data = []

# 1년당 5일만 순회
for date_str in valid_dates:
    for team_code in teams:
        print(f"📆 날짜: {date_str} | 팀: {team_code}")
        driver.get("https://www.koreabaseball.com/Player/Register.aspx")
        time.sleep(1.5)

        # 날짜 & 팀 설정
        driver.execute_script(f'document.getElementById("cphContents_cphContents_cphContents_hfSearchDate").value = "{date_str}";')
        driver.execute_script(f'document.getElementById("cphContents_cphContents_cphContents_hfSearchTeam").value = "{team_code}";')
        time.sleep(0.5)

        try:
            search_button = driver.find_element(By.ID, "cphContents_cphContents_cphContents_btnCalendarSelect")
            driver.execute_script("arguments[0].click();", search_button)
            time.sleep(2)

            # HTML 파싱
            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.select("table.tNData tbody tr")

            if not rows:
                print("등록 정보 없음 (skip)")
                continue

            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 5:
                    all_data.append({
                        "날짜": date_str,
                        "팀": team_code,
                        "이름": cols[1].text.strip(),
                        "투타유형": cols[2].text.strip(),
                        "체격": cols[4].text.strip()
                    })
        except Exception as e:
            print(f"오류 발생: {e}")

driver.quit()

# DataFrame으로 저장
df = pd.DataFrame(all_data)
csv_name = "KBO_player_register_.csv"
df.to_csv(csv_name, index=False, encoding="utf-8-sig")

# 다운로드
files.download(csv_name)
