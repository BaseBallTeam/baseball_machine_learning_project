# 크롤링, 정제된 경기/라인업/신체 데이터 등을 활용해 도루 상황별 통계를 생성합니다.
# 이닝, 상황별로 신체, 스코어, 결과 등 모든 도루 분석용 csv를 만듭니다.


import re
import pandas as pd
from collections import defaultdict
from google.colab import files

# 체격 CSV 불러오기 및 전처리
physique_df = pd.read_csv("/content/KBO_player_register_.csv")
physique_df = physique_df[physique_df["이름"].notna() & physique_df["체격"].notna()]
physique_df = physique_df[physique_df["체격"].str.contains("cm")]
physique_df["선수명"] = physique_df["이름"].astype(str).str.replace(r"\s+", "", regex=True).str.strip()
physique_df["체격"] = physique_df["체격"].astype(str).str.strip()
physique_df = physique_df.sort_values("날짜").drop_duplicates("선수명", keep="last").reset_index(drop=True)
physique_df["키"] = physique_df["체격"].str.extract(r"(\d+(?:\.\d+)?)cm").astype(float)
physique_df["몸무게"] = physique_df["체격"].str.extract(r"(\d+(?:\.\d+)?)kg").astype(float)

# 수정된 버전 - 딕셔너리 형태로 저장
physique_map = physique_df.set_index("선수명")[["키", "몸무게"]].to_dict(orient="index")

# 주자명 추출
def extract_runner_name(raw):
    if pd.isna(raw): return None
    raw = str(raw).strip()
    m = re.search(r"1루주자\s*([가-힣]+)", raw)
    if m: return m.group(1).strip()
    m = re.search(r"([가-힣]{2,4})", raw)
    return m.group(1).strip() if m else None

# 투수 손 정보
def extract_throw(val):
    if pd.isna(val): return None
    val = str(val)
    if "좌완" in val: return "좌"
    if "우완" in val or "언더" in val: return "우"
    return None

# 투수 타임라인 구성
def build_pitcher_timeline_with_starters(game_id, game_text, starters):
    lines = game_text.strip().split("\n")
    current_pitchers = {"초": starters["home"], "말": starters["away"]}
    current_half = None
    current_inning = None
    timeline = []
    for line in lines:
        line = line.strip()
        m = re.match(r"(\d+)회(초|말)", line)
        if m:
            current_inning = int(m.group(1))
            current_half = m.group(2)
            timeline.append({"inning": current_inning, "half": current_half, "투수": current_pitchers[current_half]})
        change = re.search(r"투수\s+(\S+)\s*:\s*투수\s+(\S+)", line)
        if change and current_half:
            current_pitchers[current_half] = change.group(2)
            timeline.append({"inning": current_inning, "half": current_half, "투수": change.group(2)})
    return timeline

# 도루 분석 (체격 포함)
def parse_doru_with_pitcher_timeline_and_physique(text_by_year, lineup_by_year, physique_map):
    all_results = []
    for year in [2023, 2024, 2025]:
        full_text = text_by_year[year]
        lineup_df = lineup_by_year[year].copy()
        lineup_df["게임ID"] = lineup_df["게임ID"].astype(str).str[:13]
        lineup_df["선수명"] = lineup_df["선수명"].astype(str).str.strip()
        lineup_df["포지션"] = lineup_df["포지션"].astype(str).str.strip()
        lineup_df["손"] = lineup_df["손"].astype(str).str.strip()

        gameid_text_map = defaultdict(str)
        for match in re.finditer(r"\[GAME_ID:([^\]]+)\]([\s\S]*?)(?=\[GAME_ID:|\Z)", full_text):
            gid, block = match.group(1), match.group(2)
            gameid_text_map[gid] += block

        for game_id, game_text in gameid_text_map.items():
            away_team = home_team = None
            for line in game_text.splitlines():
                if "회초" in line and "공격" in line:
                    m = re.search(r"회초\s+(.+?)\s+공격", line)
                    if m: away_team = m.group(1)
                if "회말" in line and "공격" in line:
                    m = re.search(r"회말\s+(.+?)\s+공격", line)
                    if m: home_team = m.group(1)
                if away_team and home_team:
                    break
            if not away_team or not home_team:
                continue

            filtered = lineup_df[lineup_df["게임ID"] == game_id]
            starter_pitchers = {}
            for _, row in filtered.iterrows():
                if row["포지션"] == "선발투수":
                    starter_pitchers[row["팀"]] = row["선수명"]
            starters = {
                "away": starter_pitchers.get(away_team, "알 수 없음"),
                "home": starter_pitchers.get(home_team, "알 수 없음"),
            }

            timeline = build_pitcher_timeline_with_starters(game_id, game_text, starters)
            def get_pitcher(inning, half):
                candidates = [t["투수"] for t in timeline if t["inning"] <= inning and t["half"] == half]
                return candidates[-1] if candidates else "알 수 없음"

            pitcher_throw_map = {}
            for _, row in lineup_df.iterrows():
                name = row["선수명"]
                throw = extract_throw(row["손"])
                if not throw:
                    game_date = int(game_id[:8])
                    for delta in [-1, 1, -2, 2, -3, 3]:
                        search_date = str(game_date + delta)
                        nearby_df = lineup_df[lineup_df["게임ID"].str.startswith(search_date)]
                        match_row = nearby_df[(nearby_df["포지션"] == "불펜투수") & (nearby_df["선수명"] == name)]
                        if not match_row.empty:
                            throw = extract_throw(match_row.iloc[0]["손"])
                            break
                pitcher_throw_map[name] = throw or "알 수 없음"

            away_score = home_score = 0
            current_half = "초"
            current_inning = 1
            balls = strikes = outs = 0
            balls_before_doru = strikes_before_doru = outs_before_doru = 0

            out_patterns = [
                r"삼진", r"태그아웃", r"플라이", r"땅볼", r"번트 아웃",
                r"도루실패", r"도루 아웃", r"견제사", r"실책 아웃", r"주자 아웃", r"포스아웃", r"병살", r"아웃"
            ]

            for line in game_text.splitlines():
                line = line.strip()

                if m := re.search(r"(\d+)회초", line):
                    current_inning = int(m.group(1))
                    current_half = "초"
                    outs = 0
                elif m := re.search(r"(\d+)회말", line):
                    current_inning = int(m.group(1))
                    current_half = "말"
                    outs = 0

                inning_label = f"{current_inning}"

                if re.search(r"\d+번타자\s+[가-힣]+", line):
                    balls = strikes = 0

                if re.search(r"구.*볼", line): balls = min(balls + 1, 4)
                if re.search(r"구.*파울", line) and strikes < 2: strikes += 1
                if re.search(r"구.*스트라이크", line): strikes = min(strikes + 1, 2)

                if not re.search(r"도루", line):
                    if any(re.search(pat, line) for pat in out_patterns):
                        outs = min(outs + 1, 3)
                    balls_before_doru = balls
                    strikes_before_doru = strikes
                    outs_before_doru = outs

                if "홈인" in line:
                    if current_half == "초":
                        away_score += line.count("홈인")
                    else:
                        home_score += line.count("홈인")
                if "홈런" in line:
                    if current_half == "초":
                        away_score += 1
                    else:
                        home_score += 1

                if "1루주자" in line and "도루" in line:
                    runner_name = extract_runner_name(line)
                    physique = physique_map.get(runner_name, {"키": None, "몸무게": None})
                    height = physique["키"]
                    weight = physique["몸무게"]

                    diff = (away_score - home_score) if current_half == "초" else (home_score - away_score)
                    score_diff_string = f"{diff:+d}"

                    pitcher = get_pitcher(current_inning, current_half)
                    throw = pitcher_throw_map.get(pitcher, "알 수 없음")
                    throw_val = {"우": 0, "좌": 1}.get(throw, "")
                    result_val = 1 if re.search(r"(도루.*진루|진루.*도루|도루로)", line) else 0 if re.search(r"(실패|아웃|도루사|태그아웃)", line) else ""

                    all_results.append({
                        "year": year,
                        "game_id": game_id,
                        "이닝": inning_label,
                        "스코어 상황": score_diff_string,
                        "O 카운트": outs_before_doru,
                        "B 카운트": balls_before_doru,
                        "S 카운트": strikes_before_doru,
                        "키": height,
                        "몸무게": weight,
                        "투수 손": throw_val,
                        "도루 성공 여부": result_val
                    })

    return pd.DataFrame(all_results)





# 중계 텍스트 + 라인업 CSV 미리 정의되어 있어야 함
text_by_year = {
    2023: open("/content/KBO_2023_inning_text.txt", encoding="utf-8").read(),
    2024: open("/content/KBO_2024_inning_text.txt", encoding="utf-8").read(),
    2025: open("/content/KBO_2025_inning_text.txt", encoding="utf-8").read(),
}

lineup_by_year = {
    2023: pd.read_csv("/content/KBO_2023_lineup.csv"),
    2024: pd.read_csv("/content/KBO_2024_lineup.csv"),
    2025: pd.read_csv("/content/KBO_2025_lineup.csv"),
}


# 실행
df_final = parse_doru_with_pitcher_timeline_and_physique(text_by_year, lineup_by_year, physique_map)

# 저장 및 다운로드
for year in [2023, 2024, 2025]:
    df_year = df_final[df_final["year"] == year]
    output_path = f"/content/KBO_{year}_steal.csv"
    df_year.drop(columns=["year", "game_id"]).to_csv(output_path, index=False, encoding="utf-8-sig")
    files.download(output_path)
