# Databricks notebook source
# MAGIC %md
# MAGIC # Commodity Prices Ingestion
# MAGIC
# MAGIC **Executive summary:** Ingests defense-relevant commodity prices from Yahoo Finance (yfinance) and IMF Primary Commodity Prices (PCPS) into raw and bronze; builds silver with cost-pressure metrics from yfinance data. Single notebook, no redundant commodity tables.
# MAGIC
# MAGIC **Data Sources**: yfinance (Yahoo Finance, no API key); IMF PCPS (API/CSV, fails gracefully if unavailable).
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.commodity_prices` - Raw commodity observations (yfinance + IMF), standard indicator schema
# MAGIC - `supply_chain.bronze.commodity_prices` - Cleaned with typed columns
# MAGIC - `supply_chain.silver.commodity_prices_monthly` - Monthly prices with change metrics (from yfinance only)
# MAGIC
# MAGIC **Idempotency:** Delta merge on (source, indicator_code, country_code, as_of_date).
# MAGIC
# MAGIC **Requirements:** Cluster must have outbound internet access. Install yfinance via `%pip install yfinance`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

%pip install yfinance pandas requests

# COMMAND ----------

import json
import csv
import io
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType,
)
from pyspark.sql.window import Window

# COMMAND ----------

CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.commodity_prices"
BRONZE_TABLE = f"{CATALOG}.bronze.commodity_prices"
SILVER_TABLE = f"{CATALOG}.silver.commodity_prices_monthly"

# Defense-relevant commodity tickers (Yahoo Finance)
DEFENSE_COMMODITIES = {
    'CL=F': {'name': 'Crude Oil (WTI)', 'category': 'ENERGY', 'defense_use': 'Fuel, lubricants, plastics', 'unit': 'USD/barrel'},
    'NG=F': {'name': 'Natural Gas', 'category': 'ENERGY', 'defense_use': 'Manufacturing energy, heating', 'unit': 'USD/MMBtu'},
    'GC=F': {'name': 'Gold', 'category': 'PRECIOUS_METALS', 'defense_use': 'Electronics, connectors, plating', 'unit': 'USD/oz'},
    'SI=F': {'name': 'Silver', 'category': 'PRECIOUS_METALS', 'defense_use': 'Electronics, soldering, contacts', 'unit': 'USD/oz'},
    'PL=F': {'name': 'Platinum', 'category': 'PRECIOUS_METALS', 'defense_use': 'Catalytic converters, sensors', 'unit': 'USD/oz'},
    'PA=F': {'name': 'Palladium', 'category': 'PRECIOUS_METALS', 'defense_use': 'Catalytic converters, electronics', 'unit': 'USD/oz'},
    'HG=F': {'name': 'Copper', 'category': 'INDUSTRIAL_METALS', 'defense_use': 'Wiring, motors, electronics, radiators', 'unit': 'USD/lb'},
    'ALI=F': {'name': 'Aluminum', 'category': 'INDUSTRIAL_METALS', 'defense_use': 'Vehicle frames, armor, components', 'unit': 'USD/lb'},
    'SLX': {'name': 'Steel ETF (VanEck)', 'category': 'INDUSTRIAL_METALS', 'defense_use': 'Vehicle frames, armor, structural components', 'unit': 'USD/share'},
    'LIT': {'name': 'Lithium & Battery Tech ETF', 'category': 'BATTERY_MATERIALS', 'defense_use': 'Batteries, hybrid vehicle systems', 'unit': 'USD/share'},
    'RUBUUSD': {'name': 'Rubber', 'category': 'RUBBER', 'defense_use': 'Tires, seals, gaskets', 'unit': 'USD/kg'},
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared HTTP / schema helper

# COMMAND ----------

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
# MAGIC ## Fetch yfinance (standard indicator rows)

# COMMAND ----------

import yfinance as yf

def fetch_yfinance_commodity_rows(tickers: dict, years_back: int = 5) -> list:
    """Fetch yfinance commodity prices; return list of standard indicator row dicts."""
    ingested_at = datetime.utcnow().isoformat() + "Z"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    rows = []
    for ticker, info in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                continue
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            monthly = close.resample('M').last()
            for date, price in monthly.items():
                price_val = float(price) if getattr(price, 'ndim', 0) == 0 else float(price.iloc[0])
                as_of_date = date.strftime('%Y-%m-%d')
                rows.append(normalize_indicator_row(
                    source="yfinance_commodity",
                    ingested_at=ingested_at,
                    as_of_date=as_of_date,
                    country_code="",
                    indicator_code=ticker,
                    indicator_name=info['name'],
                    value=price_val,
                    unit=info['unit'],
                    frequency="monthly",
                    raw_payload=json.dumps({"ticker": ticker, "close": price_val, "date": as_of_date}),
                ))
        except Exception as e:
            print(f"  Skip {ticker}: {e}")
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch IMF PCPS (standard indicator rows)

# MAGIC Tries API/CSV; skips if unavailable (no duplicate commodity series—yfinance is primary for silver).

# COMMAND ----------

def fetch_imf_commodity_rows() -> list:
    """Fetch IMF Primary Commodity Price indices; return list of standard indicator row dicts. Returns [] if unavailable."""
    ingested_at = datetime.utcnow().isoformat() + "Z"
    rows = []
    urls = [
        "https://api.imf.org/data/PCPS?format=jsondata",
        "https://data.imf.org/regular.aspx?key=60972224&format=csv",
    ]
    for url in urls:
        try:
            r = safe_get(url, timeout=90)
            text = r.text
            if not text or len(text) < 50:
                continue
            if "json" in url or text.strip().startswith("["):
                try:
                    data = parse_json(text)
                except Exception:
                    continue
                if isinstance(data, list):
                    for rec in data:
                        if not isinstance(rec, dict):
                            continue
                        period = rec.get("Period") or rec.get("date") or rec.get("Time Period")
                        val = rec.get("Value") or rec.get("value")
                        code = rec.get("Commodity") or rec.get("indicator") or rec.get("Indicator")
                        name = rec.get("Commodity Name") or rec.get("indicator_name") or code or "Commodity"
                        if period and val is not None:
                            try:
                                as_of = f"{str(period)[:4]}-{str(period)[4:6].zfill(2)}-01" if len(str(period)) >= 6 else f"{period}-01-01"
                            except Exception:
                                as_of = f"{period}-01-01"
                            rows.append(normalize_indicator_row(
                                source="imf_commodity_prices",
                                ingested_at=ingested_at,
                                as_of_date=as_of,
                                country_code="",
                                indicator_code=str(code) if code else "UNKNOWN",
                                indicator_name=str(name),
                                value=float(val),
                                unit=rec.get("Unit") or rec.get("unit"),
                                frequency="monthly",
                                raw_payload=json.dumps(rec),
                            ))
                    if rows:
                        return rows
            else:
                reader = csv.DictReader(io.StringIO(text))
                for rec in reader:
                    period = rec.get("Period") or rec.get("Date") or rec.get("Time Period") or ""
                    val = rec.get("Value") or rec.get("value")
                    code = rec.get("Commodity") or rec.get("Indicator") or rec.get("indicator") or ""
                    name = rec.get("Commodity Name") or rec.get("Indicator Name") or code or "Commodity"
                    if not period or val is None or str(val).strip() == "":
                        continue
                    try:
                        value_float = float(val)
                        as_of = f"{str(period)[:4]}-{str(period)[4:6].zfill(2)}-01" if len(str(period)) >= 6 else f"{period}-01-01"
                    except (TypeError, ValueError):
                        continue
                    rows.append(normalize_indicator_row(
                        source="imf_commodity_prices",
                        ingested_at=ingested_at,
                        as_of_date=as_of,
                        country_code="",
                        indicator_code=str(code) if code else "UNKNOWN",
                        indicator_name=str(name),
                        value=value_float,
                        unit=rec.get("Unit") or rec.get("unit"),
                        frequency="monthly",
                        raw_payload=json.dumps(rec),
                    ))
                if rows:
                    return rows
        except Exception as e:
            print(f"IMF URL failed: {e}")
            continue
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest: yfinance + IMF

# COMMAND ----------

all_rows = []
# yfinance (primary)
print("Fetching commodity prices from yfinance...")
yf_rows = fetch_yfinance_commodity_rows(DEFENSE_COMMODITIES, years_back=5)
if yf_rows:
    all_rows.extend(yf_rows)
    print(f"Fetched {len(yf_rows)} yfinance commodity records")
else:
    raise RuntimeError(
        "yfinance returned no data. Ensure the cluster has outbound internet access "
        "and yfinance is installed (pip install yfinance)."
    )

# IMF (additional; skip if unavailable)
imf_rows = fetch_imf_commodity_rows()
if imf_rows:
    all_rows.extend(imf_rows)
    print(f"Fetched {len(imf_rows)} IMF commodity records")
else:
    print("IMF PCPS unavailable; skipping (no redundant data).")

if not all_rows:
    raise RuntimeError("No commodity data (yfinance and IMF both failed).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Spark DataFrame and merge into raw/bronze

# COMMAND ----------

INDICATOR_SCHEMA = StructType([
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

df_raw = spark.createDataFrame(all_rows, INDICATOR_SCHEMA)
df_raw = df_raw.withColumn("ingested_at", F.col("ingested_at").cast(TimestampType()))
df_raw = df_raw.withColumn("as_of_date", F.to_date(F.col("as_of_date")))

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")

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
print(f"Raw row count after merge: {raw_count}")

# COMMAND ----------

bronze_df = spark.table(RAW_TABLE)
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
print(f"Bronze row count after merge: {bronze_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: cost-pressure metrics (yfinance only)

# MAGIC Build `commodity_prices_monthly` from bronze where source = 'yfinance_commodity'; compute pct_change and cost_pressure_score in Spark.

# COMMAND ----------

yf_bronze = spark.table(BRONZE_TABLE).filter(F.col("source") == "yfinance_commodity")
window = Window.partitionBy("indicator_code").orderBy(F.col("as_of_date"))

silver_base = yf_bronze \
    .withColumn("month", F.col("as_of_date").cast("string")) \
    .withColumn("close_price", F.col("value")) \
    .withColumn("prev_1", F.lag("value", 1).over(window)) \
    .withColumn("prev_3", F.lag("value", 3).over(window)) \
    .withColumn("prev_12", F.lag("value", 12).over(window)) \
    .withColumn("pct_change_1mo", F.when(F.col("prev_1").isNotNull() & (F.col("prev_1") != 0), (F.col("value") - F.col("prev_1")) / F.col("prev_1") * 100).otherwise(None)) \
    .withColumn("pct_change_3mo", F.when(F.col("prev_3").isNotNull() & (F.col("prev_3") != 0), (F.col("value") - F.col("prev_3")) / F.col("prev_3") * 100).otherwise(None)) \
    .withColumn("pct_change_12mo", F.when(F.col("prev_12").isNotNull() & (F.col("prev_12") != 0), (F.col("value") - F.col("prev_12")) / F.col("prev_12") * 100).otherwise(None)) \
    .withColumn("ticker", F.col("indicator_code")) \
    .withColumn("commodity_name", F.col("indicator_name")) \
    .withColumn("month_date", F.col("as_of_date")) \
    .withColumn("price_direction",
        F.when(F.col("pct_change_1mo") > 5, "RISING_FAST")
         .when(F.col("pct_change_1mo") > 0, "RISING")
         .when(F.col("pct_change_1mo") > -5, "FALLING")
         .otherwise("FALLING_FAST")) \
    .withColumn("volatility_flag",
        F.when(F.abs(F.col("pct_change_1mo")) > 10, "HIGH_VOLATILITY")
         .when(F.abs(F.col("pct_change_1mo")) > 5, "MODERATE_VOLATILITY")
         .otherwise("LOW_VOLATILITY")) \
    .withColumn("cost_pressure_score",
        F.coalesce(F.col("pct_change_3mo"), F.lit(0)) * 0.5 +
        F.coalesce(F.col("pct_change_1mo"), F.lit(0)) * 0.3 +
        F.coalesce(F.col("pct_change_12mo"), F.lit(0)) * 0.2) \
    .withColumn("ingestion_timestamp", F.current_timestamp())

meta_pd = pd.DataFrame([{"ticker": k, "category": v["category"], "defense_use": v["defense_use"]} for k, v in DEFENSE_COMMODITIES.items()])
meta_spark = spark.createDataFrame(meta_pd)
commodity_enriched = silver_base.join(meta_spark, silver_base.ticker == meta_spark.ticker, "left") \
    .select(
        silver_base["month"], silver_base["ticker"], silver_base["commodity_name"],
        F.coalesce(meta_spark["category"], F.lit("OTHER")).alias("category"),
        F.coalesce(meta_spark["defense_use"], F.lit("")).alias("defense_use"),
        silver_base["unit"], silver_base["close_price"],
        silver_base["pct_change_1mo"], silver_base["pct_change_3mo"], silver_base["pct_change_12mo"],
        silver_base["month_date"], silver_base["price_direction"], silver_base["volatility_flag"],
        silver_base["cost_pressure_score"], silver_base["ingestion_timestamp"]
    )

# COMMAND ----------

commodity_enriched.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

silver_count = spark.table(SILVER_TABLE).count()
print(f"Saved {silver_count} records to {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log row counts and sample

# COMMAND ----------

print("=== Commodity Ingestion (yfinance + IMF) ===")
print(f"Raw {RAW_TABLE}: {raw_count} rows")
print(f"Bronze {BRONZE_TABLE}: {bronze_count} rows")
print(f"Silver {SILVER_TABLE}: {silver_count} rows (yfinance only)")
display(spark.table(RAW_TABLE).orderBy(F.desc("as_of_date")).limit(10))

# COMMAND ----------

display(commodity_enriched.filter(F.col("month") >= "2024-01-01").orderBy(F.desc("month"), "category").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables: `supply_chain.raw.commodity_prices`, `supply_chain.bronze.commodity_prices`, `supply_chain.silver.commodity_prices_monthly`. Idempotent merge on (source, indicator_code, country_code, as_of_date). IMF data in raw/bronze only; silver is yfinance-only for cost-pressure metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `08_weather_ingestion` for weather risk data
# MAGIC 2. Proceed to transformation notebooks for unified demand signals
