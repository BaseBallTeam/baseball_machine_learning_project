# 연도별 KBO 경기 일정 파일을 활용, 원하는 연도별 경기 일자를 파싱 및 정리하는 코드입니다.
# csv 입력 파일을 바탕으로 경기 일정의 주요 정보를 추출합니다.


from selenium.webdriver.support.ui import Select

def get_kbo_schedule(year=2025):
    all_data = []

    driver.get("https://www.koreabaseball.com/Schedule/Schedule.aspx")
    time.sleep(3)

    try:
        year_dropdown = Select(driver.find_element(By.ID, "ddlYear"))
        year_dropdown.select_by_value(str(year))
        time.sleep(2)

        for month in range(3, 12):
            print(f"\n{year}년 {month:02d}월 스케줄 수집중")

            month_dropdown = Select(driver.find_element(By.ID, "ddlMonth"))
            month_dropdown.select_by_value(f"{month:02d}")
            time.sleep(3)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            table = soup.find("table", class_="tbl")
            if not table:
                print(f"{month}월 테이블을 찾을 수 없습니다.")
                continue

            rows = table.find_all("tr")[1:]
            current_date = None

            for row in rows:
                cols = row.find_all("td")
                texts = [col.get_text(strip=True) for col in cols]

                if len(texts) > 0 and re.match(r"\d{2}\.\d{2}", texts[0]):
                    current_date = texts[0]
                    all_data.append([month, current_date] + texts[1:])
                elif current_date:
                    all_data.append([month, current_date] + texts)

    except Exception as e:
        print(f"처리 중 오류 발생 : {e}")

    df = pd.DataFrame(all_data)
    filepath = f"/content/KBO_{year}_schedule.csv"
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"\n{year}년 스케줄 저장 완료 → {filepath}")
    return filepath

# 실행
from google.colab import files

for year in [2023, 2024, 2025]:
    file_path = get_kbo_schedule(year=year)
    files.download(file_path)