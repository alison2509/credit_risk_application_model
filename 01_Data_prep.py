import os
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------
DATA_DIR = r"C:/Users/sonng/data_analyst/Project/credit_risk_application_model/data"   # thư mục chứa các file train_*.parquet
SAMPLE_SIZE = 100000                 # lay mau 100.000 ban ghi de de dang xay dung
# ---------------------------------------------------------------------

os.makedirs("output", exist_ok=True)
OUTPUT_FILE = "output/data_flat.parquet"


# =====================================================================
# 1. Doc du lieu
# =====================================================================

base = pd.read_parquet(os.path.join(DATA_DIR, "train_base.parquet"))

print("Bang goc:", base.shape)
print("Ty le default toan bo du lieu:", round(base["target"].mean(), 4))
print("WEEK_NUM chay tu", base["WEEK_NUM"].min(), "den", base["WEEK_NUM"].max())


# =====================================================================
# 2. Lay mau 100000 ban ghi
# =====================================================================

sample_ratio = SAMPLE_SIZE / len(base)
sample_parts = []

for week, group in base.groupby("WEEK_NUM"):
    n_rows = int(round(len(group) * sample_ratio))
    if n_rows > 0:
        sample_parts.append(group.sample(n=n_rows, random_state=42)) # random_state=42 giúp kết quả tái lập (chạy lại vẫn ra cùng mẫu)

sample = pd.concat(sample_parts).reset_index(drop=True)

print()
print("So don trong mau:", len(sample))
print("Ty le default cua mau:", round(sample["target"].mean(), 4))


# =====================================================================
# 3. Doc cac bang thong tin
# =====================================================================

FILES_TO_READ = [
    "train_static_0_0.parquet",
    "train_static_0_1.parquet",
    "train_static_cb_0.parquet",
]

case_ids = set(sample["case_id"])
loaded_tables = {}

for file_name in FILES_TO_READ:
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        print("Khong tim thay file.", file_name)
        continue

    chunks = []
    parquet_file = pq.ParquetFile(file_path)

    for chunk in parquet_file.iter_batches(batch_size=100000):
        chunk = chunk.to_pandas()
        chunk = chunk[chunk["case_id"].isin(case_ids)]
        chunks.append(chunk)

    table = pd.concat(chunks, ignore_index=True)
    loaded_tables[file_name] = table
    print("Du lieu", file_name, "->", table.shape)


# =====================================================================
# 4. Union thanh bang final
# =====================================================================

data = sample
static_0_parts = []

for file_name in ["train_static_0_0.parquet", "train_static_0_1.parquet"]:
    if file_name in loaded_tables:
        static_0_parts.append(loaded_tables[file_name])

if len(static_0_parts) > 0:
    static_0 = pd.concat(static_0_parts, ignore_index=True)
    static_0 = static_0.drop_duplicates(subset="case_id")
    data = data.merge(static_0, on="case_id", how="left")

if "train_static_cb_0.parquet" in loaded_tables:
    static_cb = loaded_tables["train_static_cb_0.parquet"]
    static_cb = static_cb.drop_duplicates(subset="case_id")
    data = data.merge(static_cb, on="case_id", how="left")
    print("Final_output:", data.shape) # merge train_static_0_0 và train_static_0_1 với train_static_cb_0


# =====================================================================
# 5. Luu file
# =====================================================================

for col in data.columns:
    if data[col].dtype == "float64":
        data[col] = data[col].astype("float32")

data.to_parquet(OUTPUT_FILE, index=False)
