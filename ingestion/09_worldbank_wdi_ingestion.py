# Databricks notebook source
# MAGIC %md
# MAGIC # World Bank WDI Ingestion
# MAGIC
# MAGIC **Executive summary:** Ingests curated World Development Indicators (economic, industrial, energy, trade, inflation) from the World Bank API into raw and bronze. Used as structured external risk/demand indicators (replacing GDELT for key metrics).
# MAGIC
# MAGIC **Data Source**: https://api.worldbank.org/v2 — World Bank Indicators API (no API key). Per-indicator JSON: `country/all/indicator/{CODE}?format=json&date=...&per_page=10000`.
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.worldbank_wdi` - Raw WDI observations
# MAGIC - `supply_chain.bronze.worldbank_wdi` - Cleaned WDI with typed columns
# MAGIC
# MAGIC **Idempotency:** Delta merge on (source, indicator_code, country_code, as_of_date).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import json
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, DateType,
)

# COMMAND ----------

CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.worldbank_wdi"
BRONZE_TABLE = f"{CATALOG}.bronze.worldbank_wdi"

# Curated WDI indicators: manufacturing, trade, energy, inflation
WDI_INDICATORS = {
    "NV.IND.MANF.CD": "Manufacturing, value added (current US$)",
    "NE.EXP.GNFS.CD": "Exports of goods and services (current US$)",
    "NE.IMP.GNFS.CD": "Imports of goods and services (current US$)",
    "EG.USE.PCAP.KG.OE": "Energy use (kg of oil equivalent per capita)",
    "FP.CPI.TOTL": "Consumer price index (2010 = 100)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SL.IND.EMPL.ZS": "Employment in industry (% of total employment)",
}

WORLDBANK_BASE = "https://api.worldbank.org/v2"
DATE_RANGE = "2000:2025"
PER_PAGE = 10000

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
    # Fallback when run as notebook (no __file__)
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
# MAGIC ## Fetch WDI data from API

# COMMAND ----------

def fetch_wdi_indicator(indicator_code: str, indicator_name: str) -> list:
    """Fetch one WDI indicator for all countries; returns list of normalized row dicts."""
    url = f"{WORLDBANK_BASE}/country/all/indicator/{indicator_code}?format=json&date={DATE_RANGE}&per_page={PER_PAGE}"
    try:
        r = safe_get(url)
        payload = parse_json(r.text)
    except Exception as e:
        raise RuntimeError(f"World Bank WDI API unavailable or invalid response for {indicator_code}: {e}") from e
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    meta, data = payload[0], payload[1]
    if not isinstance(data, list):
        return []
    ingested_at = datetime.utcnow().isoformat() + "Z"
    rows = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        date_val = rec.get("date")
        if not date_val:
            continue
        try:
            as_of_date = f"{date_val}-01-01"
        except Exception:
            continue
        country = rec.get("country") or {}
        country_id = country.get("id") or ""
        country_iso = rec.get("countryiso3code") or country_id
        val = rec.get("value")
        if val is None:
            continue
        try:
            value_float = float(val)
        except (TypeError, ValueError):
            continue
        unit = (rec.get("unit") or "").strip() or None
        row = normalize_indicator_row(
            source="worldbank_wdi",
            ingested_at=ingested_at,
            as_of_date=as_of_date,
            country_code=country_iso or country_id,
            indicator_code=indicator_code,
            indicator_name=indicator_name,
            value=value_float,
            unit=unit,
            frequency="annual",
            raw_payload=json.dumps(rec),
        )
        rows.append(row)
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest all indicators

# COMMAND ----------

all_rows = []
for code, name in WDI_INDICATORS.items():
    try:
        rows = fetch_wdi_indicator(code, name)
        all_rows.extend(rows)
        print(f"Fetched {code}: {len(rows)} records")
    except Exception as e:
        print(f"Skip {code}: {e}")

if not all_rows:
    raise RuntimeError("World Bank WDI API returned no data. Check connectivity and indicator codes.")

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

# Create raw table if not exists with explicit schema
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

# Idempotent merge into raw
dt_raw = DeltaTable.forName(spark, RAW_TABLE)
dt_raw.alias("t").merge(
    df_raw.alias("s"),
    "t.source = s.source AND t.indicator_code = s.indicator_code AND t.country_code = s.country_code AND t.as_of_date = s.as_of_date"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

raw_count = spark.table(RAW_TABLE).count()
print(f"Raw table row count after merge: {raw_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze: cleaned types

# COMMAND ----------

bronze_df = spark.table(RAW_TABLE).select(
    F.col("source"),
    F.col("ingested_at"),
    F.col("as_of_date"),
    F.col("country_code"),
    F.col("indicator_code"),
    F.col("indicator_name"),
    F.col("value"),
    F.col("unit"),
    F.col("frequency"),
    F.col("raw_payload"),
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

print(f"=== World Bank WDI Ingestion ===")
print(f"Raw {RAW_TABLE}: {raw_count} rows")
print(f"Bronze {BRONZE_TABLE}: {bronze_count} rows")
print("Sample (raw):")
display(spark.table(RAW_TABLE).orderBy(F.desc("as_of_date")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables: `supply_chain.raw.worldbank_wdi`, `supply_chain.bronze.worldbank_wdi`. Idempotent merge on (source, indicator_code, country_code, as_of_date).
