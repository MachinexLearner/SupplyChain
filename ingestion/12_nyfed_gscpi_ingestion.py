# Databricks notebook source
# MAGIC %md
# MAGIC # NY Fed Global Supply Chain Pressure Index (GSCPI) Ingestion
# MAGIC
# MAGIC **Executive summary:** Ingests the NY Fed Global Supply Chain Pressure Index (monthly) into raw and bronze. Used as a structured supply chain risk indicator.
# MAGIC
# MAGIC **Data Source**: NY Fed GSCPI. Tries known CSV/JSON URLs; fails gracefully with clear message if unavailable. Data is at https://www.newyorkfed.org/research/policy/gscpi (download from page or API if exposed).
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.nyfed_gscpi` - Raw GSCPI observations
# MAGIC - `supply_chain.bronze.nyfed_gscpi` - Cleaned with typed columns
# MAGIC
# MAGIC **Idempotency:** Delta merge on (source, indicator_code, country_code, as_of_date). country_code = "" (global index).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import json
import csv
import io
import re
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, DateType,
)

# COMMAND ----------

CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.nyfed_gscpi"
BRONZE_TABLE = f"{CATALOG}.bronze.nyfed_gscpi"

# Possible GSCPI data URLs (NY Fed may change paths)
GSCPI_URLS = [
    "https://www.newyorkfed.org/medialibrary/media/research/policy/gscpi/gscpi_data.csv",
    "https://resources.newyorkfed.org/research/policy/gscpi/gscpi_data.csv",
    "https://www.newyorkfed.org/research/policy/gscpi/data/gscpi_data.csv",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared HTTP helper

# COMMAND ----------

import sys
import os
for _p in [os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "", os.getcwd(), "/Workspace/Repos", "."]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from ingestion_utils import safe_get, parse_json, normalize_indicator_row
except ImportError:
    import requests
    import time
    def safe_get(url, *, timeout=60, retries=3, backoff=2.0, headers=None):
        last = None
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=timeout, headers=headers or {})
                r.raise_for_status()
                return r
            except Exception as e:
                last = e
                if attempt < retries - 1:
                    time.sleep(backoff ** attempt)
        raise last
    def parse_json(text):
        return json.loads(text)
    def normalize_indicator_row(*, source, ingested_at, as_of_date, country_code, indicator_code, indicator_name, value, unit, frequency, raw_payload=None):
        return {"source": source, "ingested_at": ingested_at, "as_of_date": as_of_date, "country_code": country_code or "", "indicator_code": indicator_code, "indicator_name": indicator_name, "value": float(value) if value is not None else None, "unit": unit, "frequency": frequency, "raw_payload": raw_payload}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch GSCPI data
# MAGIC
# MAGIC Tries each URL; parses CSV or JSON. Fails with clear message if none work.

# COMMAND ----------

def fetch_nyfed_gscpi() -> list:
    """
    Fetch NY Fed GSCPI (monthly). Tries configured URLs; returns list of normalized row dicts.
    """
    ingested_at = datetime.utcnow().isoformat() + "Z"
    rows = []
    last_exc = None

    for url in GSCPI_URLS:
        try:
            r = safe_get(url, timeout=60)
            text = r.text.strip()
            if not text or "Page Not Found" in text or "does not exist" in text:
                continue
            # CSV: expect Date, GSCPI or similar columns
            if "\n" in text and ("," in text or "\t" in text):
                sep = "\t" if "\t" in text.split("\n")[0] else ","
                reader = csv.DictReader(io.StringIO(text), delimiter=sep)
                for rec in reader:
                    date_str = rec.get("Date") or rec.get("date") or rec.get("Month") or ""
                    val = rec.get("GSCPI") or rec.get("gscpi") or rec.get("Value") or rec.get("value")
                    if not date_str or val is None or str(val).strip() == "":
                        continue
                    try:
                        value_float = float(val)
                    except (TypeError, ValueError):
                        continue
                    # Normalize date to YYYY-MM-01
                    match = re.match(r"(\d{4})[-/]?(\d{1,2})?", str(date_str))
                    if match:
                        y, m = match.group(1), (match.group(2) or "1").zfill(2)
                        as_of_date = f"{y}-{m}-01"
                    else:
                        as_of_date = str(date_str)[:10] if len(str(date_str)) >= 10 else f"{date_str}-01-01"
                    row = normalize_indicator_row(
                        source="nyfed_gscpi",
                        ingested_at=ingested_at,
                        as_of_date=as_of_date,
                        country_code="",
                        indicator_code="GSCPI",
                        indicator_name="Global Supply Chain Pressure Index",
                        value=value_float,
                        unit="index",
                        frequency="monthly",
                        raw_payload=json.dumps(rec),
                    )
                    rows.append(row)
                if rows:
                    return rows
            # JSON
            if text.startswith("[") or text.startswith("{"):
                try:
                    data = parse_json(text)
                except Exception:
                    continue
                if isinstance(data, list):
                    for rec in data:
                        if not isinstance(rec, dict):
                            continue
                        date_str = rec.get("Date") or rec.get("date") or rec.get("Month")
                        val = rec.get("GSCPI") or rec.get("gscpi") or rec.get("value")
                        if date_str is None or val is None:
                            continue
                        try:
                            value_float = float(val)
                        except (TypeError, ValueError):
                            continue
                        as_of_date = str(date_str)[:10] if len(str(date_str)) >= 10 else f"{date_str}-01-01"
                        row = normalize_indicator_row(
                            source="nyfed_gscpi",
                            ingested_at=ingested_at,
                            as_of_date=as_of_date,
                            country_code="",
                            indicator_code="GSCPI",
                            indicator_name="Global Supply Chain Pressure Index",
                            value=value_float,
                            unit="index",
                            frequency="monthly",
                            raw_payload=json.dumps(rec),
                        )
                        rows.append(row)
                    if rows:
                        return rows
        except Exception as e:
            last_exc = e
            print(f"GSCPI URL {url[:60]}... failed: {e}")
            continue

    raise RuntimeError(
        "NY Fed GSCPI data unavailable: all configured URLs failed or returned no data. "
        "Check https://www.newyorkfed.org/research/policy/gscpi for current download option and update GSCPI_URLS in this notebook."
    ) from last_exc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

try:
    all_rows = fetch_nyfed_gscpi()
    print(f"Fetched {len(all_rows)} GSCPI records")
except RuntimeError as e:
    print(str(e))
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Spark DataFrame and schema

# COMMAND ----------

INDICATOR_RAW_SCHEMA = StructType([
    StructField("source", StringType(), False),
    StructField("ingested_at", StringType(), False),
    StructField("as_of_date", StringType(), False),
    StructField("country_code", StringType(), False),
    StructField("indicator_code", StringType(), False),
    StructField("indicator_name", StringType(), False),
    StructField("value", DoubleType(), True),
    StructField("unit", StringType(), True),
    StructField("frequency", StringType(), False),
    StructField("raw_payload", StringType(), True),
])

df_raw = spark.createDataFrame(all_rows, INDICATOR_RAW_SCHEMA)
df_raw = df_raw.withColumn("ingested_at", F.col("ingested_at").cast(TimestampType()))
df_raw = df_raw.withColumn("as_of_date", F.to_date(F.col("as_of_date")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog and idempotent merge (raw)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")
print(f"Catalog {CATALOG}, schemas raw/bronze ready.")

raw_create_sql = f"""
CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
  source STRING NOT NULL,
  ingested_at TIMESTAMP NOT NULL,
  as_of_date DATE NOT NULL,
  country_code STRING NOT NULL,
  indicator_code STRING NOT NULL,
  indicator_name STRING NOT NULL,
  value DOUBLE,
  unit STRING,
  frequency STRING NOT NULL,
  raw_payload STRING
) USING DELTA
"""
spark.sql(raw_create_sql)

# COMMAND ----------

from delta.tables import DeltaTable

dt_raw = DeltaTable.forName(spark, RAW_TABLE)
dt_raw.alias("t").merge(
    df_raw.alias("s"),
    "t.source = s.source AND t.indicator_code = s.indicator_code AND t.country_code = s.country_code AND t.as_of_date = s.as_of_date"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

raw_count = spark.table(RAW_TABLE).count()
print(f"Raw table row count after merge: {raw_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze

# COMMAND ----------

bronze_df = spark.table(RAW_TABLE).select(
    F.col("source"), F.col("ingested_at"), F.col("as_of_date"), F.col("country_code"),
    F.col("indicator_code"), F.col("indicator_name"), F.col("value"), F.col("unit"),
    F.col("frequency"), F.col("raw_payload"),
)
bronze_create_sql = f"""
CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
  source STRING NOT NULL,
  ingested_at TIMESTAMP NOT NULL,
  as_of_date DATE NOT NULL,
  country_code STRING NOT NULL,
  indicator_code STRING NOT NULL,
  indicator_name STRING NOT NULL,
  value DOUBLE,
  unit STRING,
  frequency STRING NOT NULL,
  raw_payload STRING
) USING DELTA
"""
spark.sql(bronze_create_sql)
dt_bronze = DeltaTable.forName(spark, BRONZE_TABLE)
dt_bronze.alias("t").merge(
    bronze_df.alias("s"),
    "t.source = s.source AND t.indicator_code = s.indicator_code AND t.country_code = s.country_code AND t.as_of_date = s.as_of_date"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

bronze_count = spark.table(BRONZE_TABLE).count()
print(f"Bronze table row count after merge: {bronze_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log row counts and sample

# COMMAND ----------

print("=== NY Fed GSCPI Ingestion ===")
print(f"Raw {RAW_TABLE}: {raw_count} rows")
print(f"Bronze {BRONZE_TABLE}: {bronze_count} rows")
display(spark.table(RAW_TABLE).orderBy(F.desc("as_of_date")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables: `supply_chain.raw.nyfed_gscpi`, `supply_chain.bronze.nyfed_gscpi`. Idempotent merge on (source, indicator_code, country_code, as_of_date).
