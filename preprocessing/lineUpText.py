# 저장된 라인업 텍스트 파일을 읽어 경기별 라인업 정보를 DataFrame으로 정규화합니다.
# 연도별 인원, 포지션 등의 표준 csv 라인업 파일을 만듭니다.

import pandas as pd
import os
import re
from google.colab import files

# 라인업 텍스트 파싱
def parse_lineup_text(text):
    lines = text.splitlines()
    data = {}
    team_order = []
    i = 0
    current_team = None
    current_section = None

    while i < len(lines):
        line = lines[i].strip()

        if line.endswith("선발"):
            current_team = line.replace("선발", "").strip()
            team_order.append(current_team)
            data[current_team] = {"선발투수": "", "타순": [], "후보야수": [], "불펜투수": []}
            current_section = "선발"
            i += 2
            data[current_team]["선발투수"] = lines[i].strip()
            i += 2
            continue

        elif line == "후보야수":
            current_section = "후보야수"
            i += 1
            continue
        elif line == "불펜투수":
            current_section = "불펜투수"
            i += 1
            continue
        elif current_section in ["후보야수", "불펜투수"] and line.endswith(current_section):
            current_team = line.replace(current_section, "").strip()
            i += 1
            continue

        if current_section == "선발":
            try:
                int(line)
                batter = f"{lines[i+1].strip()} ({lines[i+2].strip()})"
                data[current_team]["타순"].append(batter)
                i += 3
            except:
                i += 1
        elif current_section in ["후보야수", "불펜투수"]:
            data[current_team][current_section].append(line)
            i += 1
        else:
            i += 1

    return data, team_order

# 연도별 반복
for year in [2023, 2024, 2025]:
    rows = []

    # lineup_2023, lineup_2024, lineup_2025 중 해당 연도만 필터링
    txt_files = [f for f in os.listdir("/content") if f.startswith("lineup_") and f.endswith(".txt") and str(year) in f]

    print(f"📁 {year}년 라인업 파일 {len(txt_files)}개 처리 중")

    for file in sorted(txt_files):
        with open(f"/content/{file}", "r", encoding="utf-8") as f:
            text = f.read()
            game_id = file.replace("lineup_", "").replace(".txt", "")[:13]
            parsed_data, team_order = parse_lineup_text(text)

        for team in team_order:
            team_data = parsed_data[team]
            rows.append([game_id, team, "선발투수", team_data["선발투수"], "", ""])

            for idx, batter in enumerate(team_data["타순"], 1):
                batter_clean = re.sub(r"\(([^,]+)\s*,\s*([^)]+)\)", r"\1 \2", batter).strip()
                parts = batter_clean.split()
                rows.append([game_id, team, f"{idx}번타자"] + parts[:3])

            subs = team_data["후보야수"]
            for i in range(0, len(subs)-1, 2):
                name = subs[i]
                pos = subs[i+1].replace(",", "").strip().split()
                rows.append([game_id, team, "후보야수", name] + pos[:2])

            pens = team_data["불펜투수"]
            for i in range(0, len(pens)-1, 2):
                name = pens[i]
                pos = pens[i+1].strip()
                hand = "우완언더" if "언더" in pos else pos
                rows.append([game_id, team, "불펜투수", name, "투수", hand])

    # DataFrame으로 저장
    if rows:
        df = pd.DataFrame(rows, columns=["게임ID", "팀", "포지션", "선수명", "포지션", "손"])
        output_path = f"/content/KBO_{year}_lineup.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {output_path}")
        files.download(output_path)
    else:
        print(f"{year}년에는 처리할 파일이 없습니다.")
