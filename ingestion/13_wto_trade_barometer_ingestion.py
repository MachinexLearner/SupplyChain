# Databricks notebook source
# MAGIC %md
# MAGIC # WTO Goods Trade Barometer Ingestion
# MAGIC
# MAGIC **Executive summary:** Ingests WTO Goods Trade Barometer (quarterly) when available via API. WTO Timeseries API may require a free API key from https://apiportal.wto.org/. Fails gracefully with clear message if data is not available programmatically (no scraping).
# MAGIC
# MAGIC **Data Source**: WTO Stats/Timeseries API. Tries public endpoints first; if API key is set (e.g. WTO_API_KEY), uses it. Otherwise raises with instructions.
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.wto_trade_barometer` - Raw barometer observations
# MAGIC - `supply_chain.bronze.wto_trade_barometer` - Cleaned with typed columns
# MAGIC
# MAGIC **Idempotency:** Delta merge on (source, indicator_code, country_code, as_of_date). country_code = "" (global).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import json
import os
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, DateType,
)

# COMMAND ----------

CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.wto_trade_barometer"
BRONZE_TABLE = f"{CATALOG}.bronze.wto_trade_barometer"

# WTO API key (optional): set env WTO_API_KEY or spark.conf for programmatic access
try:
    _key = spark.conf.get("wto.api.key") if spark else ""
except Exception:
    _key = ""
WTO_API_KEY = os.environ.get("WTO_API_KEY", "") or _key or ""

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
# MAGIC ## Fetch WTO Trade Barometer
# MAGIC
# MAGIC Tries public data URL or WTO API (with key). Fails with clear message if not feasible without scraping.

# COMMAND ----------

def fetch_wto_trade_barometer() -> list:
    """
    Fetch WTO Goods Trade Barometer (quarterly). Uses public export or API with key.
    Returns list of normalized row dicts; country_code = "".
    """
    ingested_at = datetime.utcnow().isoformat() + "Z"
    rows = []

    # Try 1: Public bulk/download endpoint (if WTO exposes barometer without key)
    public_urls = [
        "https://stats.wto.org/api/v1/barometer",
        "https://api.wto.org/timeseries/v1/data",
    ]
    headers = {}
    if WTO_API_KEY:
        headers["Ocp-Apim-Subscription-Key"] = WTO_API_KEY
        headers["Authorization"] = f"Bearer {WTO_API_KEY}"

    for url in public_urls:
        try:
            r = safe_get(url, timeout=60, headers=headers if headers else None)
            text = r.text
            if not text or len(text) < 10:
                continue
            data = parse_json(text)
            # Normalize: expect list of {period, value} or {date, barometer_value} etc.
            if isinstance(data, list):
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    period = rec.get("Period") or rec.get("period") or rec.get("Date") or rec.get("date") or rec.get("Time Period")
                    val = rec.get("Value") or rec.get("value") or rec.get("Barometer") or rec.get("barometer")
                    if period is None or val is None:
                        continue
                    try:
                        value_float = float(val)
                    except (TypeError, ValueError):
                        continue
                    # Quarter to first day: e.g. 2024Q1 -> 2024-01-01
                    ps = str(period).upper()
                    if "Q" in ps:
                        parts = ps.split("Q")
                        y = parts[0][:4]
                        q = parts[1][:1] if len(parts) > 1 else "1"
                        m = {"1": "01", "2": "04", "3": "07", "4": "10"}.get(q, "01")
                        as_of_date = f"{y}-{m}-01"
                    else:
                        as_of_date = str(period)[:10] if len(str(period)) >= 10 else f"{period}-01-01"
                    row = normalize_indicator_row(
                        source="wto_trade_barometer",
                        ingested_at=ingested_at,
                        as_of_date=as_of_date,
                        country_code="",
                        indicator_code="WTO_GOODS_BAROMETER",
                        indicator_name="WTO Goods Trade Barometer",
                        value=value_float,
                        unit="index",
                        frequency="quarterly",
                        raw_payload=json.dumps(rec),
                    )
                    rows.append(row)
                if rows:
                    return rows
            if isinstance(data, dict):
                # Single value or nested
                for k, v in data.items():
                    if k in ("metadata", "Meta", "info"):
                        continue
                    if isinstance(v, (int, float)):
                        row = normalize_indicator_row(
                            source="wto_trade_barometer",
                            ingested_at=ingested_at,
                            as_of_date=datetime.utcnow().strftime("%Y-%m-%d"),
                            country_code="",
                            indicator_code="WTO_GOODS_BAROMETER",
                            indicator_name="WTO Goods Trade Barometer",
                            value=float(v),
                            unit="index",
                            frequency="quarterly",
                            raw_payload=json.dumps(data),
                        )
                        rows.append(row)
                        return rows
        except Exception as e:
            print(f"WTO URL {url[:50]}... failed: {e}")
            continue

    # No data without scraping
    raise RuntimeError(
        "WTO Goods Trade Barometer data is not available programmatically with current configuration. "
        "Options: (1) Sign up for a free API key at https://apiportal.wto.org/ and set WTO_API_KEY (env or spark.conf 'wto.api.key'), "
        "(2) Download data manually from https://www.wto.org/english/res_e/statis_e/wtoi_e.htm and load into this table, "
        "(3) Skip this notebook if WTO barometer is not required."
    ) from None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

try:
    all_rows = fetch_wto_trade_barometer()
    print(f"Fetched {len(all_rows)} WTO Trade Barometer records")
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

print("=== WTO Trade Barometer Ingestion ===")
print(f"Raw {RAW_TABLE}: {raw_count} rows")
print(f"Bronze {BRONZE_TABLE}: {bronze_count} rows")
display(spark.table(RAW_TABLE).orderBy(F.desc("as_of_date")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables: `supply_chain.raw.wto_trade_barometer`, `supply_chain.bronze.wto_trade_barometer`. Idempotent merge on (source, indicator_code, country_code, as_of_date).
