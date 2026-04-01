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

# ── Cài thêm tiện ích hệ thống ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────
# pyspark KHÔNG được cài lại — bitnami đã cung cấp Spark + PySpark
# qua PYTHONPATH /opt/bitnami/spark/python
COPY requirements.txt /tmp/requirements.txt
RUN /opt/bitnami/python/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/bitnami/python/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# ── Hadoop S3A JARs (cần để đọc/ghi s3a://... vào MinIO) ─────────
# Phiên bản khớp với Hadoop 3.3.4 bundled trong bitnami/spark:3.5
ENV HADOOP_VERSION=3.3.4
ENV AWS_SDK_VERSION=1.12.262
ENV SPARK_JARS_DIR=/opt/bitnami/spark/jars

RUN curl -fSL \
    "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" \
    -o "${SPARK_JARS_DIR}/hadoop-aws-${HADOOP_VERSION}.jar" \
  && curl -fSL \
    "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" \
    -o "${SPARK_JARS_DIR}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar"

# ── Source code & config ──────────────────────────────────────────
WORKDIR /app
RUN mkdir -p data/logs .cache/huggingface

COPY src/     ./src/
COPY config/  ./config/
COPY scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# ── Biến môi trường mặc định ─────────────────────────────────────
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    HOME=/app \
    USER=sparkuser \
    HADOOP_USER_NAME=sparkuser

RUN if ! getent passwd 1001 >/dev/null; then \
      echo 'sparkuser:x:1001:0:Spark User:/app:/bin/bash' >> /etc/passwd; \
    fi \
    && chown -R 1001:0 /app

USER 1001
