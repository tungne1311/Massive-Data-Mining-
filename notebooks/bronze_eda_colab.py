# %% [markdown]
# # Bronze EDA tren Google Colab bang DuckDB + Hugging Face Hub
#
# Notebook nay doc truc tiep cac file Parquet trong repo Hugging Face:
# `chuongdo1104/amazon-2023-bronze`
#
# Phan tich:
# - Tong so user, item, interactions cho train/val/test.
# - Gom user tren ca 3 split de tinh frequency/user.
# - Tinh item frequency theo train va all splits.
# - Dem user/item co tan suat <= 1, <= 2, <= 3, <= 5...
# - Goi y nguong k-core.
# - Tach HEAD/MID/TAIL theo train_freq bang fixed threshold, quantile, Pareto.

# %%
!pip -q install "duckdb>=1.0.0" "huggingface_hub>=0.23.0" pandas pyarrow

# %%
import json
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import HfApi, hf_hub_url

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)

REPO_ID = "chuongdo1104/amazon-2023-bronze"
REPO_TYPE = "dataset"
BRANCH = "main"

# Neu repo private, mo comment 2 dong duoi va nhap token trong Colab.
# from google.colab import userdata
# HF_TOKEN = userdata.get("HF_TOKEN")
HF_TOKEN = None

# %%
api = HfApi(token=HF_TOKEN)
repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, revision=BRANCH)

def parquet_urls(prefix: str) -> list[str]:
    files = sorted(
        f for f in repo_files
        if f.startswith(prefix.rstrip("/") + "/") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"Khong tim thay parquet files trong prefix: {prefix}")
    return [
        hf_hub_url(repo_id=REPO_ID, filename=f, repo_type=REPO_TYPE, revision=BRANCH)
        for f in files
    ]

train_urls = parquet_urls("bronze/bronze_train.parquet")
val_urls = parquet_urls("bronze/bronze_val.parquet")
test_urls = parquet_urls("bronze/bronze_test.parquet")

print("So part files:")
print("train:", len(train_urls))
print("val:  ", len(val_urls))
print("test: ", len(test_urls))
print("Vi du URL:", train_urls[0])

# %%
con = duckdb.connect(database=":memory:")
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA memory_limit='10GB';")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

if HF_TOKEN:
    # Cho private dataset tren Hugging Face.
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET http_retries=5;")
    con.execute("SET custom_user_agent='duckdb-colab-bronze-eda';")

# %%
def create_split_view(view_name: str, urls: list[str]) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            reviewer_id,
            parent_asin,
            rating,
            timestamp,
            helpful_vote
        FROM read_parquet(?);
        """,
        [urls],
    )

create_split_view("train", train_urls)
create_split_view("val", val_urls)
create_split_view("test", test_urls)

con.execute(
    """
    CREATE OR REPLACE VIEW interactions_all AS
    SELECT 'train' AS split, * FROM train
    UNION ALL
    SELECT 'val' AS split, * FROM val
    UNION ALL
    SELECT 'test' AS split, * FROM test;
    """
)

# Materialize 2 bang frequency de cac query sau nhanh hon.
con.execute(
    """
    CREATE OR REPLACE TEMP TABLE user_freq AS
    SELECT
        reviewer_id,
        count(*) AS n_interactions,
        count(*) FILTER (WHERE split = 'train') AS n_train,
        count(*) FILTER (WHERE split = 'val') AS n_val,
        count(*) FILTER (WHERE split = 'test') AS n_test,
        count(DISTINCT parent_asin) AS n_unique_items
    FROM interactions_all
    GROUP BY reviewer_id;
    """
)

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE item_freq AS
    SELECT
        parent_asin,
        count(*) AS n_interactions,
        count(*) FILTER (WHERE split = 'train') AS n_train,
        count(*) FILTER (WHERE split = 'val') AS n_val,
        count(*) FILTER (WHERE split = 'test') AS n_test,
        count(DISTINCT reviewer_id) AS n_unique_users
    FROM interactions_all
    GROUP BY parent_asin;
    """
)

print("Da tao views/tables: train, val, test, interactions_all, user_freq, item_freq")

# %%
def q(sql: str) -> pd.DataFrame:
    return con.sql(sql).df()

def show(title: str, sql: str) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    df = q(sql)
    display(df)
    return df

results = {}

# %% [markdown]
# ## 1. Tong quan train/val/test

# %%
results["split_overview"] = show(
    "Split overview",
    """
    SELECT
        split,
        count(*) AS interactions,
        count(DISTINCT reviewer_id) AS users,
        count(DISTINCT parent_asin) AS items,
        min(timestamp) AS min_ts,
        max(timestamp) AS max_ts
    FROM interactions_all
    GROUP BY split
    ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END;
    """,
)

results["all_overview"] = show(
    "All splits overview",
    """
    SELECT
        count(*) AS interactions,
        count(DISTINCT reviewer_id) AS users,
        count(DISTINCT parent_asin) AS items,
        count(DISTINCT reviewer_id || '|' || parent_asin) AS unique_user_item_pairs
    FROM interactions_all;
    """,
)

# %% [markdown]
# ## 2. Phan phoi so interactions/user tren ca train + val + test

# %%
results["user_freq_summary"] = show(
    "User frequency summary - all splits",
    """
    SELECT
        count(*) AS users,
        min(n_interactions) AS min_freq,
        quantile_cont(n_interactions, 0.25) AS p25,
        quantile_cont(n_interactions, 0.50) AS p50,
        quantile_cont(n_interactions, 0.75) AS p75,
        quantile_cont(n_interactions, 0.90) AS p90,
        quantile_cont(n_interactions, 0.95) AS p95,
        quantile_cont(n_interactions, 0.99) AS p99,
        max(n_interactions) AS max_freq,
        avg(n_interactions) AS avg_freq
    FROM user_freq;
    """,
)

results["user_lte_counts"] = show(
    "So luong user co interactions <= k",
    """
    WITH cuts(k) AS (VALUES (1), (2), (3), (4), (5), (10), (20), (50), (100))
    SELECT
        k AS max_interactions,
        count(*) FILTER (WHERE n_interactions <= k) AS users_lte_k,
        round(100.0 * count(*) FILTER (WHERE n_interactions <= k) / count(*), 4) AS pct_users_lte_k
    FROM cuts
    CROSS JOIN user_freq
    GROUP BY k
    ORDER BY k;
    """,
)

# %% [markdown]
# ## 3. Phan phoi so interactions/item

# %%
results["item_freq_summary_all"] = show(
    "Item frequency summary - all splits",
    """
    SELECT
        count(*) AS items,
        min(n_interactions) AS min_freq,
        quantile_cont(n_interactions, 0.25) AS p25,
        quantile_cont(n_interactions, 0.50) AS p50,
        quantile_cont(n_interactions, 0.75) AS p75,
        quantile_cont(n_interactions, 0.90) AS p90,
        quantile_cont(n_interactions, 0.95) AS p95,
        quantile_cont(n_interactions, 0.99) AS p99,
        max(n_interactions) AS max_freq,
        avg(n_interactions) AS avg_freq
    FROM item_freq;
    """,
)

results["item_train_freq_summary"] = show(
    "Item train_freq summary - chi tinh n_train > 0",
    """
    SELECT
        count(*) FILTER (WHERE n_train > 0) AS train_items,
        min(n_train) FILTER (WHERE n_train > 0) AS min_train_freq,
        quantile_cont(n_train, 0.50) FILTER (WHERE n_train > 0) AS p50,
        quantile_cont(n_train, 0.80) FILTER (WHERE n_train > 0) AS p80,
        quantile_cont(n_train, 0.90) FILTER (WHERE n_train > 0) AS p90,
        quantile_cont(n_train, 0.95) FILTER (WHERE n_train > 0) AS p95,
        quantile_cont(n_train, 0.99) FILTER (WHERE n_train > 0) AS p99,
        max(n_train) AS max_train_freq
    FROM item_freq;
    """,
)

results["item_lte_counts"] = show(
    "So luong item co interactions <= k tren all splits",
    """
    WITH cuts(k) AS (VALUES (1), (2), (3), (4), (5), (10), (20), (50), (100), (500), (1000))
    SELECT
        k AS max_interactions,
        count(*) FILTER (WHERE n_interactions <= k) AS items_lte_k,
        round(100.0 * count(*) FILTER (WHERE n_interactions <= k) / count(*), 4) AS pct_items_lte_k
    FROM cuts
    CROSS JOIN item_freq
    GROUP BY k
    ORDER BY k;
    """,
)

results["item_train_lte_counts"] = show(
    "So luong item co train_freq <= k, bo qua cold-in-train",
    """
    WITH cuts(k) AS (VALUES (1), (2), (3), (4), (5), (10), (20), (50), (100), (500), (1000)),
    train_items AS (
        SELECT * FROM item_freq WHERE n_train > 0
    )
    SELECT
        k AS max_train_freq,
        count(*) FILTER (WHERE n_train <= k) AS items_lte_k,
        round(100.0 * count(*) FILTER (WHERE n_train <= k) / count(*), 4) AS pct_items_lte_k
    FROM cuts
    CROSS JOIN train_items
    GROUP BY k
    ORDER BY k;
    """,
)

# %% [markdown]
# ## 4. Bang goi y nguong k-core
#
# Luu y: bang nay la uoc tinh 1-pass theo frequency hien tai. K-core that su la iterative filter:
# sau khi remove user/item, degree cua phia con lai thay doi. Bang nay dung de chon nguong khoi dau.

# %%
results["user_core_candidates"] = show(
    "User-core candidates",
    """
    WITH thresholds(k) AS (VALUES (2), (3), (4), (5), (10), (20), (50), (100))
    SELECT
        k,
        count(*) FILTER (WHERE u.n_interactions < k) AS users_removed_if_user_core_k,
        count(*) FILTER (WHERE u.n_interactions >= k) AS users_kept_if_user_core_k,
        sum(u.n_interactions) FILTER (WHERE u.n_interactions < k) AS interactions_lost_by_removed_users,
        round(100.0 * count(*) FILTER (WHERE u.n_interactions < k) / count(*), 4) AS pct_users_removed
    FROM thresholds
    CROSS JOIN user_freq u
    GROUP BY k
    ORDER BY k;
    """,
)

results["item_core_candidates"] = show(
    "Item-core candidates",
    """
    WITH thresholds(k) AS (VALUES (2), (3), (4), (5), (10), (20), (50), (100))
    SELECT
        k,
        count(*) FILTER (WHERE i.n_interactions < k) AS items_removed_if_item_core_k,
        count(*) FILTER (WHERE i.n_interactions >= k) AS items_kept_if_item_core_k,
        sum(i.n_interactions) FILTER (WHERE i.n_interactions < k) AS interactions_lost_by_removed_items,
        round(100.0 * count(*) FILTER (WHERE i.n_interactions < k) / count(*), 4) AS pct_items_removed
    FROM thresholds
    CROSS JOIN item_freq i
    GROUP BY k
    ORDER BY k;
    """,
)

# %% [markdown]
# ## 5. Tach HEAD / MID / TAIL cho item
#
# Nen dung `n_train` de phan loai popularity, tranh leakage tu val/test.

# %%
results["head_mid_tail_fixed"] = show(
    "HEAD/MID/TAIL fixed thresholds by train_freq",
    """
    SELECT
        CASE
            WHEN n_train = 0 THEN 'COLD_IN_TRAIN'
            WHEN n_train <= 10 THEN 'TAIL_1_10'
            WHEN n_train <= 50 THEN 'MID_11_50'
            ELSE 'HEAD_51_PLUS'
        END AS popularity_group,
        count(*) AS items,
        sum(n_train) AS train_interactions,
        sum(n_val) AS val_interactions,
        sum(n_test) AS test_interactions
    FROM item_freq
    GROUP BY 1
    ORDER BY
        CASE popularity_group
            WHEN 'HEAD_51_PLUS' THEN 1
            WHEN 'MID_11_50' THEN 2
            WHEN 'TAIL_1_10' THEN 3
            ELSE 4
        END;
    """,
)

results["head_mid_tail_quantile"] = show(
    "HEAD/MID/TAIL quantile by train_freq",
    """
    WITH train_items AS (
        SELECT parent_asin, n_train
        FROM item_freq
        WHERE n_train > 0
    ),
    q AS (
        SELECT
            quantile_cont(n_train, 0.80) AS p80,
            quantile_cont(n_train, 0.95) AS p95
        FROM train_items
    )
    SELECT
        CASE
            WHEN t.n_train >= q.p95 THEN 'HEAD_GE_P95'
            WHEN t.n_train >= q.p80 THEN 'MID_P80_TO_P95'
            ELSE 'TAIL_LT_P80'
        END AS popularity_group,
        count(*) AS items,
        min(t.n_train) AS min_train_freq,
        max(t.n_train) AS max_train_freq,
        sum(t.n_train) AS train_interactions
    FROM train_items t
    CROSS JOIN q
    GROUP BY 1
    ORDER BY
        CASE popularity_group
            WHEN 'HEAD_GE_P95' THEN 1
            WHEN 'MID_P80_TO_P95' THEN 2
            ELSE 3
        END;
    """,
)

results["pareto_item_train_thresholds"] = show(
    "Pareto thresholds: top item theo train_freq can bao nhieu item de cover X% train interactions",
    """
    WITH ranked AS (
        SELECT
            parent_asin,
            n_train,
            sum(n_train) OVER () AS total_train_interactions,
            sum(n_train) OVER (ORDER BY n_train DESC, parent_asin) AS cum_train_interactions,
            row_number() OVER (ORDER BY n_train DESC, parent_asin) AS rank_by_freq,
            count(*) OVER () AS total_train_items
        FROM item_freq
        WHERE n_train > 0
    ),
    targets(target_interaction_share) AS (VALUES (0.50), (0.80), (0.90), (0.95))
    SELECT
        target_interaction_share,
        min(rank_by_freq) AS items_needed,
        round(100.0 * min(rank_by_freq) / max(total_train_items), 4) AS pct_items_needed,
        min(n_train) AS min_train_freq_in_head
    FROM ranked
    JOIN targets ON cum_train_interactions >= target_interaction_share * total_train_interactions
    GROUP BY target_interaction_share
    ORDER BY target_interaction_share;
    """,
)

# %% [markdown]
# ## 6. Cold-start item trong val/test so voi train

# %%
results["val_test_cold_items"] = show(
    "Val/Test item cold-start vs train",
    """
    SELECT
        split,
        count(*) AS interactions,
        count(*) FILTER (WHERE i.n_train = 0) AS cold_item_interactions,
        round(100.0 * count(*) FILTER (WHERE i.n_train = 0) / count(*), 4) AS pct_cold_item_interactions,
        count(DISTINCT a.parent_asin) AS items,
        count(DISTINCT a.parent_asin) FILTER (WHERE i.n_train = 0) AS cold_items
    FROM interactions_all a
    JOIN item_freq i USING (parent_asin)
    WHERE split IN ('val', 'test')
    GROUP BY split
    ORDER BY split;
    """,
)

# %% [markdown]
# ## 7. Top users/items de inspect head

# %%
results["top_items_train_freq"] = show(
    "Top 50 items by train_freq",
    """
    SELECT parent_asin, n_train, n_val, n_test, n_interactions
    FROM item_freq
    ORDER BY n_train DESC, parent_asin
    LIMIT 50;
    """,
)

results["top_users_all_splits"] = show(
    "Top 50 users by all-split interactions",
    """
    SELECT reviewer_id, n_interactions, n_train, n_val, n_test, n_unique_items
    FROM user_freq
    ORDER BY n_interactions DESC, reviewer_id
    LIMIT 50;
    """,
)

# %% [markdown]
# ## 8. Export ket qua

# %%
out_dir = Path("/content/bronze_eda_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

for name, df in results.items():
    df.to_csv(out_dir / f"{name}.csv", index=False)

json_ready = {name: df.to_dict(orient="records") for name, df in results.items()}
(out_dir / "bronze_eda_summary.json").write_text(
    json.dumps(json_ready, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

print("Da luu outputs vao:", out_dir)
print("Files:")
for p in sorted(out_dir.iterdir()):
    print("-", p.name)

