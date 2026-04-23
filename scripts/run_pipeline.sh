#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# run_pipeline.sh — Spark-submit wrapper cho RecSys Pipeline
#
# Cách dùng:
#   scripts/run_pipeline.sh --step 1_2        # Bronze: Ingestion + Processing
#   scripts/run_pipeline.sh --step 3_silver   # Silver: Text profiles + Enrichment
#   scripts/run_pipeline.sh --step 4_gold     # Gold: ID mapping + Embeddings
#   scripts/run_pipeline.sh --all             # Full: Bronze → Silver → Gold
#   scripts/run_pipeline.sh --help
#
# Tất cả args được pass thẳng vào pipeline_runner.py
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SPARK_HOME="${SPARK_HOME:-/opt/bitnami/spark}"
SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"
export HOME="${HOME:-/app}"
export USER="${USER:-sparkuser}"
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-sparkuser}"
# ── Cluster config ────────────────────────────────────────────────
SPARK_MASTER="${SPARK_MASTER:-spark://spark-master:7077}"
DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-4g}"
EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-4g}"
EXECUTOR_CORES="${SPARK_EXECUTOR_CORES:-2}"

# ── MinIO / S3A config ────────────────────────────────────────────
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://minio:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

# ── Driver host: dùng IP của container này để workers gửi kết quả về ─
# Trong Docker network (bridge), container IP là routable với các container khác
DRIVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -z "$DRIVER_IP" ]; then
  DRIVER_IP="127.0.0.1"
fi

CONFIG_FILE="${CONFIG_FILE:-/app/config/config.yaml}"

echo "========================================================"
echo "  RecSys Pipeline — spark-submit"
echo "  Master     : ${SPARK_MASTER}"
echo "  Driver IP  : ${DRIVER_IP}"
echo "  Driver mem : ${DRIVER_MEMORY}"
echo "  Executor   : ${EXECUTOR_MEMORY} / ${EXECUTOR_CORES} cores"
echo "  Config     : ${CONFIG_FILE}"
echo "  Args       : $*"
echo "========================================================"

exec "$SPARK_SUBMIT" \
  --master "$SPARK_MASTER" \
  --deploy-mode client \
  --driver-memory "$DRIVER_MEMORY" \
  --executor-memory "$EXECUTOR_MEMORY" \
  --executor-cores "$EXECUTOR_CORES" \
  \
  --conf "spark.driver.bindAddress=0.0.0.0" \
  --conf "spark.driver.host=${DRIVER_IP}" \
  \
  --conf "spark.sql.adaptive.enabled=true" \
  --conf "spark.sql.adaptive.coalescePartitions.enabled=true" \
  --conf "spark.sql.adaptive.skewJoin.enabled=true" \
  \
  --conf "spark.executorEnv.PYTHONPATH=/app/src" \
  --conf "spark.executorEnv.PYTHONUNBUFFERED=1" \
  --conf "spark.executorEnv.TOKENIZERS_PARALLELISM=false" \
  --conf "spark.executorEnv.HF_HOME=/app/.cache/huggingface" \
  --conf "spark.executorEnv.HOME=/app" \
  --conf "spark.executorEnv.USER=sparkuser" \
  --conf "spark.executorEnv.HADOOP_USER_NAME=sparkuser" \
  --conf "spark.python.worker.reuse=true" \
  \
  --conf "spark.hadoop.fs.s3a.endpoint=${MINIO_ENDPOINT}" \
  --conf "spark.hadoop.fs.s3a.access.key=${MINIO_ACCESS_KEY}" \
  --conf "spark.hadoop.fs.s3a.secret.key=${MINIO_SECRET_KEY}" \
  --conf "spark.hadoop.fs.s3a.path.style.access=true" \
  --conf "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" \
  --conf "spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider" \
  --conf "spark.hadoop.fs.s3a.connection.ssl.enabled=false" \
  --conf "spark.hadoop.fs.s3a.fast.upload=true" \
  --conf "spark.hadoop.fs.s3a.multipart.size=104857600" \
  \
  /app/src/pipeline_runner.py \
  --config "$CONFIG_FILE" \
  "$@"
