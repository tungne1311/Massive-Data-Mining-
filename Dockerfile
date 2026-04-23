# ─────────────────────────────────────────────────────────────────
# RecSys Pipeline v3 — Cluster Edition
#
# Base: bitnami/spark:3.5  (Spark 3.5.x + Python 3.11 + Hadoop 3.3.x)
#
# Image này dùng chung cho 3 role:
#   spark-master  → SPARK_MODE=master
#   spark-worker  → SPARK_MODE=worker  SPARK_MASTER_URL=spark://spark-master:7077
#   pipeline      → chạy run_pipeline.sh (spark-submit client mode)
# ─────────────────────────────────────────────────────────────────
FROM bitnamilegacy/spark:3.5.2

USER root

# ── Biến môi trường hệ thống ─────────────────────────────────────
ENV HADOOP_VERSION=3.3.4 \
    AWS_SDK_VERSION=1.12.262 \
    SPARK_JARS_DIR=/opt/bitnami/spark/jars \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    HOME=/app \
    USER=sparkuser \
    HADOOP_USER_NAME=sparkuser

WORKDIR /app

# ── GỘP CHUNG 1 LAYER DUY NHẤT ĐỂ TỐI ƯU DUNG LƯỢNG IMAGE ──────────
# 1. Cài đặt các gói hệ thống và tải file jar
# 2. Cài python requirements
# 3. Dọn dẹp bộ nhớ đệm ẩn và gỡ các công cụ trung gian (curl)
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && /opt/bitnami/python/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/bitnami/python/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && curl -fSL "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" -o "${SPARK_JARS_DIR}/hadoop-aws-${HADOOP_VERSION}.jar" \
    && curl -fSL "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" -o "${SPARK_JARS_DIR}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/requirements.txt /root/.cache

# ── Tạo định danh user & Cấu trúc thư mục ──────────────────────
# Cấp quyền ngay tại tạo thư mục
RUN if ! getent passwd 1001 >/dev/null; then \
      echo 'sparkuser:x:1001:0:Spark User:/app:/bin/bash' >> /etc/passwd; \
    fi \
    && mkdir -p data/logs .cache/huggingface \
    && chown -R 1001:0 /app

# ── COPY SOURCE KÈM QUYỀN SỞ HỮU TRỰC TIẾP ───────────────────────
# Dùng --chown ngay lúc COPY để không bị lặp layer (dung lượng X2 file source)
COPY --chown=1001:0 src/ ./src/
COPY --chown=1001:0 config/ ./config/
COPY --chown=1001:0 scripts/ ./scripts/

RUN chmod +x ./scripts/*.sh

USER 1001
