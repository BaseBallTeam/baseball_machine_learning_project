# 도루 관련 데이터를 전처리 후 XGBoost 분류 모델로 도루 성공/실패 예측을 수행합니다.
# 모델 학습, 검증, 2025년 도루 성공 확률 csv 예측파일을 생성하는 코드입니다.


from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 파일 업로드
uploaded = files.upload()  # KBO_2023_steal.csv, KBO_2024_steal.csv, KBO_2025_steal.csv 선택

# 파일 불러오기
df_2023 = pd.read_csv("KBO_2023_steal.csv")
df_2024 = pd.read_csv("KBO_2024_steal.csv")
df_2025 = pd.read_csv("KBO_2025_steal.csv")

# 학습용 데이터 통합
df_train = pd.concat([df_2023, df_2024], ignore_index=True)

# 타깃 컬럼
target_col = "도루 성공 여부"

# 전처리 함수 (수치형만 사용)
def preprocess_numeric(df, is_train=True):
    df = df.copy()

    # 필요한 수치형 feature 목록
    feature_cols = ["이닝", "스코어 상황", "O 카운트", "B 카운트", "S 카운트", "키", "몸무게", "투수 손"]

    # 타입 강제 변환
    df["키"] = df["키"].astype(float)
    df["몸무게"] = df["몸무게"].astype(float)
    df["이닝"] = df["이닝"].astype(int)
    df["스코어 상황"] = df["스코어 상황"].astype(int)
    df["O 카운트"] = df["O 카운트"].astype(int)
    df["B 카운트"] = df["B 카운트"].astype(int)
    df["S 카운트"] = df["S 카운트"].astype(int)
    df["투수 손"] = df["투수 손"].astype(int)

    # 입력(X), 타깃(y) 분리
    if is_train or (target_col in df.columns):
        X = df[feature_cols]
        y = df[target_col]
        return X, y
    else:
        return df[feature_cols]  # 테스트셋 (타깃 없는 경우)

# 전처리
X, y = preprocess_numeric(df_train, is_train=True)
X_2025 = preprocess_numeric(df_2025, is_train=False)

# 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# XGBoost 모델 학습
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 검증 정확도 출력
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))


# 예측 전에 혹시 남아있는 예측 컬럼 제거
df_2025 = df_2025.drop(columns=["도루 성공 여부 예측"], errors="ignore")

# 수치형 입력값만 추출
features = ["이닝", "스코어 상황", "O 카운트", "B 카운트", "S 카운트", "키", "몸무게", "투수 손"]
X_2025 = df_2025[features].copy()

# 타입 강제 변환
for col in features:
    if col in ["키", "몸무게"]:
        X_2025[col] = X_2025[col].astype(float)
    else:
        X_2025[col] = X_2025[col].astype(int)

# 예측
pred_2025 = model.predict(X_2025)
df_2025["도루 성공 여부 예측"] = pred_2025

# 2025 예측 수행
pred_2025 = model.predict(X_2025)
df_2025["도루 성공 여부 예측"] = pred_2025

# 결과 저장
output_path = "KBO_2025_steal_predicted.csv"
df_2025.to_csv(output_path, index=False, encoding="utf-8-sig")
files.download(output_path)
