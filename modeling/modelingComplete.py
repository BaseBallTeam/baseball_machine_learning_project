# =========================================================
# KBO 도루 예측 v11 (KFold + 가중치 + Calibration + F1기반 threshold)
# =========================================================

import numpy as np
import pandas as pd
import re
from itertools import product

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support     # ★ 추가
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# 0) Load
# -----------------------------
df23 = pd.read_csv("KBO_2023_steal.csv")
df24 = pd.read_csv("KBO_2024_steal.csv")
df25 = pd.read_csv("KBO_2025_steal.csv")

TARGET   = "도루 성공 여부"
NUM_BASE = ["이닝","스코어 상황","O 카운트","B 카운트","S 카운트","주력","투수 손"]
CAT_BASE = ["키_구간","몸무게_구간"]

# -----------------------------
# 1) 전처리 + 파생 + 디자인 행렬
# -----------------------------
def coerce_keep_nan(df):
    d = df.copy()
    for c in NUM_BASE:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
    if TARGET in d:
        d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce").fillna(0).astype(int)
    for c in CAT_BASE:
        if c in d:
            d[c] = d[c].astype("string")
    return d

def add_base_feats(df):
    d = df.copy()
    d["BS_index"] = d.get("B 카운트",0)*4 + d.get("S 카운트",0)
    d["O_mul_B"]  = d.get("O 카운트",0)*d.get("B 카운트",0)
    d["O_mul_S"]  = d.get("O 카운트",0)*d.get("S 카운트",0)
    d["is_late_inning"] = (d.get("이닝",0) >= 7).astype(int)

    if "스코어 상황" in d:
        diff = d["스코어 상황"].clip(-10,10)
        bins   = [-99, -3, -1, 0, 1, 3, 99]
        labels = ["<=-3","-2~-1","0","1","2~3",">=4"]
        d["score_bucket"] = pd.cut(diff, bins=bins, labels=labels, include_lowest=True)
    else:
        d["score_bucket"] = pd.Series(pd.Categorical([np.nan]*len(d)))

    b = d.get("B 카운트", pd.Series([np.nan]*len(d)))
    s = d.get("S 카운트", pd.Series([np.nan]*len(d)))
    d["BS_combo"] = "B"+b.fillna(-1).astype(int).astype(str)+"S"+s.fillna(-1).astype(int).astype(str)
    return d

def add_speed_enhancers(df):
    d = df.copy()
    if "주력" in d:
        d["spd_x_inning"] = d["주력"]*d.get("이닝",0)
        d["spd_x_score"]  = d["주력"]*d.get("스코어 상황",0)
        d["spd_x_hand"]   = d["주력"]*d.get("투수 손",0).fillna(0)
        d["spd_bin"] = pd.cut(
            d["주력"],
            bins=[-np.inf,5.0,6.0,7.0,np.inf],
            labels=["저속","보통","빠름","매우빠름"]
        )
    return d

def add_hand_unknown(df):
    d = df.copy()
    if "투수 손" in d:
        unk = d["투수 손"].isna()
        d["is_hand_unknown"] = (unk.mean() >= 0.02).astype(int)*unk.astype(int)
    return d

def prep(df):
    for f in (coerce_keep_nan, add_base_feats, add_speed_enhancers, add_hand_unknown):
        df = f(df)
    return df

df23, df24, df25 = map(prep, [df23, df24, df25])

NUM_COLS = [c for c in (
    NUM_BASE + ["BS_index","O_mul_B","O_mul_S","is_late_inning",
                "spd_x_inning","spd_x_score","spd_x_hand","is_hand_unknown"]
) if c in df23.columns or c in df24.columns or c in df25.columns]

CAT_COLS = [c for c in (
    CAT_BASE + ["score_bucket","BS_combo","spd_bin"]
) if c in df23.columns or c in df24.columns or c in df25.columns]

def onehot(df, cols):
    if not cols: return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cols], prefix=cols, prefix_sep="=", dtype=int, dummy_na=True)

tmp_train = pd.concat([df23, df24], ignore_index=True)
X_num_tr = tmp_train[NUM_COLS].copy()
X_cat_tr = onehot(tmp_train, CAT_COLS)
SCHEMA = list(X_num_tr.columns) + list(X_cat_tr.columns)

def design(df, schema_cols, with_target=True):
    X_num = df[NUM_COLS].copy()
    X_cat = onehot(df, CAT_COLS)
    X = pd.concat([X_num, X_cat], axis=1).reindex(columns=schema_cols, fill_value=0)
    y = df[TARGET].astype(int) if (with_target and TARGET in df) else None
    X.columns = [re.sub(r'[\[\]<>]', '_', str(c)) for c in X.columns]
    X.columns = [re.sub(r'\s+', '_', c) for c in X.columns]
    return X.astype("float32"), y

X23, y23 = design(df23, SCHEMA, True)
X24, y24 = design(df24, SCHEMA, True)
X25, y25 = design(df25, SCHEMA, True)

# -----------------------------
# 2) train/valid split (23+24)
# -----------------------------
X_234 = pd.concat([X23, X24], axis=0).reset_index(drop=True)
y_234 = pd.concat([y23, y24], axis=0).reset_index(drop=True)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_234, y_234, test_size=0.2,
    random_state=42, stratify=y_234
)

fail_weight = 1.0
w_tr = np.where(y_tr.values==0, fail_weight, 1.0).astype("float32")

print(f"\nTrain size: {len(X_tr)} | Valid size: {len(X_val)} | Test(2025): {len(X25)}")

# -----------------------------
# 3) KFold 기반 파라미터 최적화
# -----------------------------
grid = {
    "learning_rate": [0.01,0.02,0.03,0.06],
    "max_depth": [3,4],
    "min_child_weight": [2,3],
    "subsample": [0.8,0.9],
    "colsample_bytree": [0.8,0.9],
    "scale_pos_weight": [1.0,1.2]
}
base_params = dict(
    objective="binary:logistic",
    eval_metric="auc",
    reg_lambda=1.0,
    max_delta_step=1,
    tree_method="hist",
    random_state=42,
    n_estimators=2000
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_acc, best_params = -1.0, None

for pr in product(*[grid[k] for k in grid]):
    params = dict(zip(grid.keys(), pr))
    acc_list = []

    for tr_idx, va_idx in kf.split(X_tr):
        X_tr2, X_va2 = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr2, y_va2 = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        w_tr2 = np.where(y_tr2.values==0, fail_weight, 1.0)

        clf = XGBClassifier(**base_params, **params)
        clf.fit(X_tr2, y_tr2, sample_weight=w_tr2, verbose=False)
        pred = clf.predict(X_va2)
        acc_list.append(accuracy_score(y_va2, pred))

    m_acc = np.mean(acc_list)
    if m_acc > best_acc:
        best_acc = m_acc
        best_params = params

print("\nBest KFold-CV params:", best_params, "| mean ACC:", round(best_acc,4))

# -----------------------------
# 4) best 파라미터로 전체 train 재학습
# -----------------------------
best_model = XGBClassifier(**base_params, **best_params)
best_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

# -----------------------------
# 5) Calibration(sigmoid)
# -----------------------------
cal_sig = CalibratedClassifierCV(
    estimator=XGBClassifier(**base_params, **best_params),
    cv=5,
    method="sigmoid"
)
cal_sig.fit(X_tr, y_tr)

# raw/sigmoid 각각 2025 확률
probas_25 = {
    "raw"    : best_model.predict_proba(X25)[:,1],
    "sigmoid": cal_sig.predict_proba(X25)[:,1]
}

# -----------------------------
# 6) 평가 함수 (ACC / BalAcc / macroF1 / 실패F1 기준 threshold)
# -----------------------------
ths = np.linspace(0.05, 0.95, 181)   # 0.05 ~ 0.95

def evaluate(tag, y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)

    acc_list      = []
    bal_list      = []
    f1_macro_list = []

    for t in ths:
        pred = (y_prob >= t).astype(int)
        acc_list.append(accuracy_score(y_true, pred))
        bal_list.append(balanced_accuracy_score(y_true, pred))

        # 0,1 클래스 각각 F1 계산
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0,1], zero_division=0
        )
        f1_macro_list.append((f1[0] + f1[1]) / 2.0)

    # 각 기준별 best threshold
    t_acc      = float(ths[int(np.argmax(acc_list))])
    t_bal      = float(ths[int(np.argmax(bal_list))])
    t_f1_macro = float(ths[int(np.argmax(f1_macro_list))])

    def report(name, thr):
        pred = (y_prob >= thr).astype(int)
        print(f"\n=== 2025 [{tag}] {name} @thr={thr:.3f} ===")
        print("AUC:", round(auc,4), "| ACC:", round(accuracy_score(y_true,pred),4))
        print(classification_report(y_true, pred, digits=4, zero_division=0))
        print("Confusion:\n", confusion_matrix(y_true, pred))

    print(f"\n>>> 2025 [{tag}] ROC-AUC = {auc:.4f}")
    report("ACC-max",            t_acc)
    report("BalAcc-max",         t_bal)
    report("macroF1-max",        t_f1_macro)

    return dict(
        auc=auc,
        t_acc=t_acc,
        t_bal=t_bal,
        t_f1_macro=t_f1_macro,
    )

scores = {tag: evaluate(tag, y25, prob) for tag, prob in probas_25.items()}


# -----------------------------
# 7) CSV 저장
# -----------------------------
print("\n==== Summary (2025) ====")
for tag, sc in scores.items():
    print(
        f"{tag}: AUC={sc['auc']:.4f} | "
        f"t_acc={sc['t_acc']:.3f} | "
        f"t_bal={sc['t_bal']:.3f} | "
        f"t_f1_macro={sc['t_f1_macro']:.3f} | "
    )

for tag, proba in probas_25.items():
    out = df25.copy()
    sc = scores[tag]

    out[f"도루 확률({tag})"]          = np.round(proba,4)
    out[f"도루 예측({tag}|ACC)"]      = (proba>=sc["t_acc"]).astype(int)
    out[f"도루 예측({tag}|BalAcc)"]   = (proba>=sc["t_bal"]).astype(int)
    out[f"도루 예측({tag}|F1macro)"]  = (proba>=sc["t_f1_macro"]).astype(int)
    out[f"도루 예측({tag}|0.700)"]    = (proba>=0.7).astype(int)

    path = f"/content/KBO_2025_pred_{tag}_v11.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved:", path)

# =========================================
# 앙상블 모델링 3가지 version 적용
# =========================================

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    precision_score, recall_score
)

# =========================================
# 0) 전제: X_tr, X_val, X25, y_tr, y_val, y25 준비돼 있다고 가정
# =========================================

# 전체 피처 컬럼 리스트 (순서 꼭 맞춰줘)
ALL_FEATURES = [
    "이닝", "스코어 상황", "O 카운트", "B 카운트", "S 카운트",
    "주력", "투수 손", "키_구간", "몸무게_구간"
]

# X가 numpy 배열이면 DataFrame으로 변환
X_tr_df  = pd.DataFrame(X_tr,  columns=ALL_FEATURES)
X_val_df = pd.DataFrame(X_val, columns=ALL_FEATURES)
X25_df   = pd.DataFrame(X25,   columns=ALL_FEATURES)

print(f"Train size: {len(X_tr_df)} | Valid size: {len(X_val_df)} | Test(2025): {len(X25_df)}")

# =========================================
# 1) 피처 그룹 정의
# =========================================
# 주자(런너) 관련 피처
FEAT_RUNNER = ["주력", "키_구간", "몸무게_구간"]

# 경기 상황 관련 피처
FEAT_SITUATION = ["이닝", "O 카운트", "B 카운트", "S 카운트", "스코어 상황"]

# 전체 피처 (런너 + 상황 + 투수)
FEAT_ALL = ALL_FEATURES  # 그냥 전부 사용

feature_groups = {
    "runner": FEAT_RUNNER,
    "situation": FEAT_SITUATION,
    "all": FEAT_ALL,
}

# runner 모델에 가중치 더 주는 형태 (임의 설정)
ensemble_weights = {
    "runner": 0.5,
    "situation": 0.2,
    "all": 0.3,
}

# =========================================
# 2) XGBoost 공통 파라미터
#    (네가 찾은 best params를 반영한 버전 예시)
# =========================================
base_params = dict(
    objective      = "binary:logistic",
    eval_metric    = "logloss",
    tree_method    = "hist",
    random_state   = 42,
    n_estimators   = 500,
    learning_rate  = 0.01,
    max_depth      = 3,
    min_child_weight = 3,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    scale_pos_weight = 1.2,  # 클래스 불균형 보정
)

fail_weight = 1.0  # 필요하면 1.2, 1.5 등으로 조정 가능

models = {}
val_probas = {}
test_probas = {}

# =========================================
# 3) 그룹별 개별 모델 학습
# =========================================
for name, cols in feature_groups.items():
    print(f"\n[Train] group = {name}, cols = {cols}")

    X_tr_g  = X_tr_df[cols].values
    X_val_g = X_val_df[cols].values
    X_te_g  = X25_df[cols].values

    # 실패(0) 클래스에만 fail_weight 적용
    w_tr = np.where(y_tr.values == 0, fail_weight, 1.0)

    clf = XGBClassifier(**base_params)
    clf.fit(
        X_tr_g, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val_g, y_val)],
        verbose=False
    )

    proba_val = clf.predict_proba(X_val_g)[:, 1]
    proba_te  = clf.predict_proba(X_te_g)[:, 1]

    models[name] = clf
    val_probas[name] = proba_val
    test_probas[name] = proba_te

    auc_val = roc_auc_score(y_val, proba_val)
    print(f"  -> [VAL] ROC-AUC = {auc_val:.4f}")

# =========================================
# 4) 가중 평균(Weighted Soft Voting) 앙상블
# =========================================
# w_runner * p_runner + w_situation * p_situation + w_all * p_all

weights_arr = np.array([ensemble_weights[k] for k in feature_groups.keys()])

val_mat = np.vstack([val_probas[k] for k in feature_groups.keys()])  # (3, N_val)
te_mat  = np.vstack([test_probas[k] for k in feature_groups.keys()])  # (3, N_test)

proba_val_ens = np.average(val_mat, axis=0, weights=weights_arr)
proba_te_ens  = np.average(te_mat,  axis=0, weights=weights_arr)

auc_val_ens = roc_auc_score(y_val, proba_val_ens)
auc_te_ens  = roc_auc_score(y25,   proba_te_ens)

print("\n=====================================================")
print(f"[Feature-ensemble] Validation ROC-AUC = {auc_val_ens:.4f}")
print(f"[Feature-ensemble] 2025 Test ROC-AUC = {auc_te_ens:.4f}")

# =========================================
# 5) (네가 썼던 threshold 탐색 코드랑 연결)
#    - proba_val_ens, proba_te_ens를 가지고
#      eval_on_thresholds(...) 같은 함수에 넣어서 thr 찾으면 됨
# =========================================
thr = 0.73
y25_pred = (proba_te_ens >= thr).astype(int)

print(f"\n=== 2025 [Feature-ensemble] @thr={thr:.3f} ===")
print("ACC:", accuracy_score(y25, y25_pred),
      "| BalACC:", balanced_accuracy_score(y25, y25_pred),
      "| REC1(성공):", recall_score(y25, y25_pred, pos_label=1),
      "| PREC1(성공):", precision_score(y25, y25_pred, pos_label=1))
print("\nClassification Report:")
print(classification_report(y25, y25_pred))
print("Confusion:")
print(confusion_matrix(y25, y25_pred))