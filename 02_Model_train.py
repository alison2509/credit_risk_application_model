import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

INPUT_FILE = "output/data_flat.parquet"
TRAIN_RATIO = 0.8
N_BINS = 5
N_FEATURES = 20

os.makedirs("output", exist_ok=True)


# ---- 1. Doc du lieu ----
print("1. Doc du lieu")
data = pd.read_parquet(INPUT_FILE)
print("So dong va so cot:", data.shape)
print(data[["case_id", "WEEK_NUM", "target"]].head())

print("Ty le khach vo no:", round(data["target"].mean(), 4))
print("So cot thieu tren 50%:", (data.isna().mean() > 0.5).sum())
print()

# ---- 2. Lam sach ----
print("2. Lam sach")
decision_date = pd.to_datetime(data["date_decision"])

for col in data.columns:
    if col != "date_decision" and str(data[col].dtype).startswith("datetime"):
        data[col] = (pd.to_datetime(data[col]) - decision_date).dt.days

y = data["target"]
week = data["WEEK_NUM"]

X = data.drop(columns=["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"])

X = X.select_dtypes(include=["number"])
print("So cot dang so:", X.shape[1])

# Bo cot thieu qua nhieu va cot chi co mot gia tri
X = X.loc[:, X.isna().mean() < 0.95] # loai bo nhung cot thieu > 95 %
X = X.loc[:, X.nunique() > 1] # loai bo nhung cot chi co 1 gia tri (khong co y nghia)
print("Con lai sau khi loc:", X.shape[1])
print()

# ---- 3. Chia train va valid theo tuan ----
print("3. Chia train va valid theo tuan")
cut_week = week.quantile(TRAIN_RATIO)

X_train = X[week <= cut_week]
X_valid = X[week > cut_week]
y_train = y[week <= cut_week]
y_valid = y[week > cut_week]
week_valid = week[week > cut_week]

print("Train:", X_train.shape, "| ty le xau:", round(y_train.mean(), 4))
print("Valid:", X_valid.shape, "| ty le xau:", round(y_valid.mean(), 4))
print()

# ---- 4. Tinh WOE va IV ----
print("4. Tinh WOE va IV")

def tinh_woe_iv(bins, y):
    df = pd.DataFrame({"bin": bins.values, "y": y.values})
    bang = df.groupby("bin")["y"].agg(["count", "sum"])

    bad = bang["sum"]
    good = bang["count"] - bang["sum"]

    pct_bad = bad / bad.sum()
    pct_good = good / good.sum()

    pct_bad = pct_bad.replace(0, 0.0001)
    pct_good = pct_good.replace(0, 0.0001)

    woe = np.log(pct_good / pct_bad)
    iv = ((pct_good - pct_bad) * woe).sum()
    return woe, iv


woe_dict = {}
iv_dict = {}
edges_dict = {}

for col in X.columns:
    if X_train[col].notna().sum() < 100:
        continue

    edges = pd.qcut(X_train[col], N_BINS, retbins=True, duplicates="drop")[1]
    if len(edges) < 3:
        continue

    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bin = pd.cut(X_train[col], bins=edges).astype(str)
    train_bin = train_bin.fillna("MISSING")
    train_bin = train_bin.replace("nan", "MISSING")

    woe, iv = tinh_woe_iv(train_bin, y_train)

    woe_dict[col] = woe
    iv_dict[col] = iv
    edges_dict[col] = edges

iv_series = pd.Series(iv_dict).sort_values(ascending=False)
print("10 bien co IV cao nhat:")
print(iv_series.head(10).round(4))
print()

# ---- 5. Chon bien va doi sang WOE ----
print("5. Chon bien va doi sang WOE")
top_features = iv_series.head(N_FEATURES).index.tolist()

X_train_woe = pd.DataFrame(index=X_train.index)
X_valid_woe = pd.DataFrame(index=X_valid.index)

for col in top_features:
    edges = edges_dict[col]
    woe = woe_dict[col]

    train_bin = pd.cut(X_train[col], bins=edges).astype(str)
    train_bin = train_bin.fillna("MISSING")
    train_bin = train_bin.replace("nan", "MISSING")

    valid_bin = pd.cut(X_valid[col], bins=edges).astype(str)
    valid_bin = valid_bin.fillna("MISSING")
    valid_bin = valid_bin.replace("nan", "MISSING")

    X_train_woe[col] = train_bin.map(woe).fillna(0).values
    X_valid_woe[col] = valid_bin.map(woe).fillna(0).values


print("Du lieu dua vao mo hinh:", X_train_woe.shape)
print()

# ---- 6. Train Logistic Regression ----
print("6. Train Logistic Regression")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_woe, y_train)

pred_train = model.predict_proba(X_train_woe)[:, 1]
pred_valid = model.predict_proba(X_valid_woe)[:, 1]

coef = pd.DataFrame({
    "feature": top_features,
    "iv": [round(iv_series[c], 4) for c in top_features],
    "coefficient": model.coef_[0].round(4),
})
print(coef)
coef.to_csv("output/model_coefficients.csv", index=False)
print()

# ---- 7. Danh gia ----
print("7. Danh gia")
def tinh_ks(y_true, pred):
    df = pd.DataFrame({"y": np.array(y_true), "p": np.array(pred)})
    df = df.sort_values("p")
    cum_bad = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    cum_good = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return (cum_bad - cum_good).abs().max()


auc_train = roc_auc_score(y_train, pred_train)
auc_valid = roc_auc_score(y_valid, pred_valid)
gini_valid = 2 * auc_valid - 1
ks_valid = tinh_ks(y_valid, pred_valid)

# Tinh Gini cho tung tuan de xem mo hinh co yeu dan theo thoi gian khong
weeks = []
ginis = []

for w in sorted(week_valid.unique()):
    mask = (week_valid == w).values
    if y_valid[mask].nunique() > 1 and mask.sum() >= 100:
        auc_w = roc_auc_score(y_valid[mask], pred_valid[mask])
        weeks.append(w)
        ginis.append(2 * auc_w - 1)

slope = np.polyfit(weeks, ginis, 1)[0]

print("AUC train:", round(auc_train, 4))
print("AUC valid:", round(auc_valid, 4))
print("Gini valid:", round(gini_valid, 4))
print("KS valid:", round(ks_valid, 4))
print("Gini trung binh theo tuan:", round(np.mean(ginis), 4))
print("Do doc:", round(slope, 5), "(am nghia la mo hinh yeu dan)")

ket_qua = pd.DataFrame([{
    "auc_train": round(auc_train, 4),
    "auc_valid": round(auc_valid, 4),
    "gini_valid": round(gini_valid, 4),
    "ks_valid": round(ks_valid, 4),
    "mean_weekly_gini": round(np.mean(ginis), 4),
    "slope": round(slope, 5),
}])
ket_qua.to_csv("output/evaluation_results.csv", index=False)
print()

# ---- 8. Doi xac suat sang diem ----
print("8. Doi xac suat sang diem")
# Cong thuc quy doi diem cua scorecard, tham khao tai lieu credit scoring.
# 600 diem ung voi ty le tot/xau la 50:1, them 20 diem thi ty le nay gap doi.
PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50

factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

odds = (1 - pred_valid) / pred_valid
score = offset + factor * np.log(odds)

print("Diem trung binh khach tra no tot:", round(score[y_valid.values == 0].mean(), 1))
print("Diem trung binh khach vo no:", round(score[y_valid.values == 1].mean(), 1))

diem = pd.DataFrame({
    "default_probability": pred_valid.round(5),
    "score": score.round(1),
    "actual_target": y_valid.values,
})
diem.to_csv("output/scorecard_points.csv", index=False)


# ---- 9. Ve bieu do ----

plt.figure(figsize=(10, 4))
plt.plot(weeks, ginis, marker="o")
plt.xlabel("WEEK_NUM")
plt.ylabel("Gini")
plt.title("Gini theo tung tuan")
plt.grid(alpha=0.3)
plt.savefig("output/stability_by_week.png", dpi=120, bbox_inches="tight")

plt.figure(figsize=(8, 6))
top_iv = iv_series.head(20).sort_values()
plt.barh(top_iv.index, top_iv.values)
plt.xlabel("IV")
plt.title("Top 20 bien theo IV")
plt.savefig("output/top_features_iv.png", dpi=120, bbox_inches="tight")

plt.figure(figsize=(9, 4))
plt.hist(score[y_valid.values == 0], bins=40, alpha=0.6, label="Tra no tot", density=True)
plt.hist(score[y_valid.values == 1], bins=40, alpha=0.6, label="Vo no", density=True)
plt.xlabel("Diem")
plt.legend()
plt.title("Phan bo diem")
plt.savefig("output/score_distribution.png", dpi=120, bbox_inches="tight")

print()