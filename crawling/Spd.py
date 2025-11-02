# 한국어 선수명, 주력수치

import pandas as pd
import re
from google.colab import files


file_list = ["2023 주루 수치.CSV", "2024 주루 수치.CSV", "2025 주루 수치.CSV"]

for file in file_list:
    try:
        year_match = re.search(r'(\d{4})', file)
        if not year_match:
            print(f"Skipping {file}: 'YYYY' 형식의 연도를 찾을 수 없습니다.")
            continue

        year = year_match.group(1)

        print(f"\n파일 처리 중: {file} ({year}년)")

        df = pd.read_csv(file, usecols=["Name", "Spd"], encoding='cp949')

        # 한국어 이름 추출
        df['선수명'] = df['Name'].str.split('?').str[-1]

        df.rename(columns={'Spd': '주력수치'}, inplace=True)
        processed_df = df[['선수명', '주력수치']]
        processed_df.dropna(inplace=True)
        output_filename = f"{year}_주력수치_processed.csv"
        processed_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"'{output_filename}' 파일 생성 완료. 다운로드를 시작합니다.")
        files.download(output_filename)

    except FileNotFoundError:
         print(f"!!! 오류: '{file}'을(를) 찾을 수 없습니다.")
         print("!!! 파일이 코랩 환경의 '/content/' 경로에 업로드되었는지 확인하세요.")
    except Exception as e:
        print(f"파일 {file} 처리 중 오류 발생: {e}")

print("\n완료")

"""/content/drive/MyDrive/기학기팀플자료/2023 주루 수치.CSV"""