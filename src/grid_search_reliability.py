"""
Grid Search Reliability Weights (Bước 3–6)

Hạ tầng chạy grid search nhỏ:
  - Loop qua từng config trong reliability_tuning.configs
  - Chạy bước 3 → 4 → 5 → 6 cho từng config_id
  - Ghi output tách riêng theo config_id
  - Tổng hợp grid_search_summary.json + .csv

KHÔNG đánh giá HR@10 / NDCG@10 ở giai đoạn này.
downstream_eval_status = "pending_step7_10" cho tất cả config.

Cách dùng:
  from grid_search_reliability import run_grid_search, run_single_config
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

import step3_silver_cleaning as step3
import step4_labeling as step4
import step5_temporal_split as step5
import step6_feature_store as step6

logger = logging.getLogger(__name__)


# ── Chạy một config ───────────────────────────────────────────────────────────

def run_single_config(
    spark: SparkSession,
    cfg: dict,
    rel_cfg: dict,
) -> dict:
    """
    Chạy bước 3→4→5→6 cho một reliability config.
    Trả về dict kết quả để tổng hợp vào grid summary.
    """
    config_id = rel_cfg["config_id"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Chạy config_id: {config_id}")
    logger.info(f"  Weights: w_v={rel_cfg['w_verified']} | "
                f"w_t={rel_cfg['w_text']} | w_h={rel_cfg['w_helpful']} | "
                f"unverified={rel_cfg['unverified_value']}")
    logger.info(f"{'='*60}")

    result = {
        "config_id":              config_id,
        "w_verified":             rel_cfg["w_verified"],
        "w_text":                 rel_cfg["w_text"],
        "w_helpful":              rel_cfg["w_helpful"],
        "unverified_value":       rel_cfg["unverified_value"],
        "downstream_eval_status": "pending_step7_10",
        "error":                  None,
    }

    try:
        # ── Bước 3: Silver Cleaning ───────────────────────────────────
        df_silver = step3.run(spark, cfg, rel_cfg, config_id)
        silver_summary = _load_silver_summary(cfg, config_id)
        result["silver_total_rows"]         = silver_summary.get("total_rows", -1)
        result["silver_avg_reliability"]    = silver_summary.get("avg_reliability_score", -1)
        result["silver_verified_ratio"]     = silver_summary.get("verified_ratio", -1)
        result["silver_short_review_ratio"] = silver_summary.get("short_review_ratio", -1)
        result["silver_dup_text_ratio"]     = silver_summary.get("duplicate_text_ratio", -1)

        # ── Bước 4: Labeling ─────────────────────────────────────────
        df_labeled = step4.run(spark, cfg, config_id, df_silver=df_silver)

        # ── Bước 5: Temporal Split ────────────────────────────────────
        split_info = step5.run(spark, cfg, config_id, df_labeled=df_labeled)
        stats      = split_info["stats"]
        result["train_rows"]  = stats.get("train", {}).get("n_rows", -1)
        result["val_rows"]    = stats.get("val",   {}).get("n_rows", -1)
        result["test_rows"]   = stats.get("test",  {}).get("n_rows", -1)
        result["train_users"] = stats.get("train", {}).get("n_users", -1)
        result["train_items"] = stats.get("train", {}).get("n_items", -1)
        result["train_months"]= "|".join(split_info.get("train_months", []))
        result["val_months"]  = "|".join(split_info.get("val_months",   []))
        result["test_months"] = "|".join(split_info.get("test_months",  []))

        # ── Bước 6: Feature Store ─────────────────────────────────────
        feat_info = step6.run(spark, cfg, config_id)
        feat_sum  = feat_info["summary"]
        result["feature_n_users"] = feat_sum.get("n_user_rows", -1)
        result["feature_n_items"] = feat_sum.get("n_item_rows", -1)
        result["feature_n_ui"]    = feat_sum.get("n_ui_rows",   -1)

        logger.info(f"  ✓ config_id={config_id} hoàn tất")

    except Exception as e:
        logger.error(f"  ✗ config_id={config_id} lỗi: {e}", exc_info=True)
        result["error"] = str(e)

    return result


def _load_silver_summary(cfg: dict, config_id: str) -> dict:
    """Đọc silver summary JSON đã ghi ở bước 3."""
    fpath = Path(cfg["paths"]["log_dir"]) / f"silver_summary_{config_id}.json"
    if fpath.exists():
        return json.loads(fpath.read_text())
    return {}


# ── Hook placeholder cho downstream evaluation ───────────────────────────────

def downstream_evaluation_hook(
    spark: SparkSession,
    cfg: dict,
    config_id: str,
    result: dict,
) -> dict:
    """
    TODO (Bước 7–10): Cắm vào đây sau khi có model trained.

    Ví dụ:
        model = train_bpr(spark, cfg, config_id)
        hr10, ndcg10 = evaluate_model(model, cfg, config_id)
        result["HR@10"]    = hr10
        result["NDCG@10"]  = ndcg10
        result["downstream_eval_status"] = "done"

    Hiện tại chỉ trả về result gốc với status pending.
    """
    logger.info(f"  [Hook] downstream_evaluation_hook cho {config_id} — chờ bước 7-10")
    result["downstream_eval_status"] = "pending_step7_10"
    return result


# ── Grid Search ───────────────────────────────────────────────────────────────

def run_grid_search(
    spark: SparkSession,
    cfg: dict,
    config_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Chạy grid search.
    config_ids: nếu None thì chạy tất cả configs trong reliability_tuning.
    Trả về list kết quả từng config.
    """
    all_configs = cfg["reliability_tuning"]["configs"]

    if config_ids is not None:
        configs_to_run = [c for c in all_configs if c["config_id"] in config_ids]
    else:
        configs_to_run = all_configs

    if not configs_to_run:
        logger.warning("Không có config nào để chạy!")
        return []

    logger.info(f"=== Grid Search: {len(configs_to_run)} config(s) ===")
    logger.info(f"  Config IDs: {[c['config_id'] for c in configs_to_run]}")

    all_results: list[dict] = []

    for rel_cfg in configs_to_run:
        result = run_single_config(spark, cfg, rel_cfg)
        # Cắm hook (hiện rỗng, để sau gắn step 7-10)
        result = downstream_evaluation_hook(spark, cfg, rel_cfg["config_id"], result)
        all_results.append(result)

    # ── Ghi summary tổng hợp ─────────────────────────────────────────
    _write_grid_summary(all_results, cfg)

    logger.info("=== Grid Search hoàn tất ===")
    return all_results


# ── Ghi summary ───────────────────────────────────────────────────────────────

def _write_grid_summary(results: list[dict], cfg: dict) -> None:
    """Ghi grid_search_summary.json và .csv vào data/logs/."""
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = log_dir / f"grid_search_summary_{ts}.json"
    payload = {
        "run_time":     datetime.now().isoformat(),
        "n_configs":    len(results),
        "note":         "downstream_eval_status=pending_step7_10 — HR@10/NDCG@10 chưa có",
        "results":      results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"  Grid summary JSON → {json_path}")

    # CSV (tiện so sánh nhanh)
    if results:
        csv_path = log_dir / f"grid_search_summary_{ts}.csv"
        all_keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"  Grid summary CSV  → {csv_path}")

    # Upload lên MinIO
    _upload_grid_summary_minio(payload, cfg, ts)


def _upload_grid_summary_minio(payload: dict, cfg: dict, ts: str) -> None:
    """Upload grid summary lên MinIO (không crash nếu lỗi)."""
    try:
        import boto3
        from botocore.client import Config

        mn = cfg["minio"]
        s3 = boto3.client(
            "s3",
            endpoint_url          = mn["endpoint"],
            aws_access_key_id     = mn["access_key"],
            aws_secret_access_key = mn["secret_key"],
            config                = Config(signature_version="s3v4"),
            region_name           = "us-east-1",
        )
        content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        key     = f"silver/logs/grid_search_summary_{ts}.json"
        s3.put_object(
            Bucket=mn["bucket"], Key=key,
            Body=content, ContentType="application/json"
        )
        logger.info(f"  Grid summary MinIO → s3a://{mn['bucket']}/{key}")
    except Exception as e:
        logger.warning(f"  Không upload grid summary lên MinIO: {e}")
