# Databricks notebook source
# MAGIC %md
# MAGIC # Tariff and Trade Risk Data Ingestion (Live API)
# MAGIC
# MAGIC **Executive summary:** Fetches tariff and trade disruption events from the **Federal Register API** into raw, silver, and gold (monthly trade risk index). Searches for tariff, trade restriction, export control, sanctions, and import duty documents published by relevant agencies.
# MAGIC
# MAGIC **Data Source**: Federal Register API — https://www.federalregister.gov/developers/documentation/api/v1 (free, no API key required)
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.trade_tariff_events` — Raw tariff/trade events
# MAGIC - `supply_chain.silver.trade_tariff_risk_events` — Processed trade risk events
# MAGIC - `supply_chain.gold.trade_tariff_risk_monthly` — Monthly trade risk scores
# MAGIC
# MAGIC **How it works:** Queries Federal Register for documents matching trade/tariff keywords from trade-related agencies, then classifies each document into an event type and severity.

# COMMAND ----------

# MAGIC %pip install requests pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import re
import time
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# Widget for configurable start year
dbutils.widgets.text("start_year", "2020", "Start year for trade event search")

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.trade_tariff_events"
SILVER_TABLE = f"{CATALOG}.silver.trade_tariff_risk_events"
GOLD_TABLE = f"{CATALOG}.gold.trade_tariff_risk_monthly"

# HTS (Harmonized Tariff Schedule) codes relevant to defense manufacturing
DEFENSE_HTS_CODES = {
    # Vehicles
    '8701': 'TRACTORS',
    '8702': 'MOTOR VEHICLES FOR TRANSPORT OF 10+ PERSONS',
    '8703': 'MOTOR VEHICLES FOR TRANSPORT OF PERSONS',
    '8704': 'MOTOR VEHICLES FOR TRANSPORT OF GOODS',
    '8705': 'SPECIAL PURPOSE MOTOR VEHICLES',
    '8706': 'CHASSIS FITTED WITH ENGINES',
    '8707': 'BODIES FOR MOTOR VEHICLES',
    '8708': 'PARTS AND ACCESSORIES OF MOTOR VEHICLES',
    
    # Armor and Protection
    '7308': 'STRUCTURES OF IRON OR STEEL',
    '7326': 'OTHER ARTICLES OF IRON OR STEEL',
    '9306': 'BOMBS, GRENADES, AMMUNITION',
    '9305': 'PARTS OF WEAPONS',
    
    # Electronics
    '8471': 'AUTOMATIC DATA PROCESSING MACHINES',
    '8517': 'TELEPHONE SETS AND TRANSMISSION APPARATUS',
    '8525': 'TRANSMISSION APPARATUS FOR RADIO/TV',
    '8526': 'RADAR, RADIO NAVIGATION APPARATUS',
    '8529': 'PARTS FOR RADIO/TV APPARATUS',
    '8536': 'ELECTRICAL APPARATUS FOR SWITCHING',
    '8544': 'INSULATED WIRE AND CABLE',
    
    # Raw Materials
    '7201': 'PIG IRON',
    '7206': 'IRON AND NON-ALLOY STEEL',
    '7208': 'FLAT-ROLLED IRON OR STEEL',
    '7219': 'STAINLESS STEEL FLAT-ROLLED',
    '7601': 'UNWROUGHT ALUMINUM',
    '7606': 'ALUMINUM PLATES AND SHEETS',
    '7403': 'REFINED COPPER',
    
    # Rubber/Tires
    '4011': 'NEW PNEUMATIC TIRES OF RUBBER',
    '4012': 'RETREADED OR USED PNEUMATIC TIRES',
    '4013': 'INNER TUBES OF RUBBER',
}

# Countries with significant trade risk for defense supply chain
TRADE_RISK_COUNTRIES = {
    'CN': ('CHINA', 'HIGH'),
    'RU': ('RUSSIA', 'CRITICAL'),
    'MX': ('MEXICO', 'MODERATE'),
    'DE': ('GERMANY', 'LOW'),
    'JP': ('JAPAN', 'LOW'),
    'KR': ('SOUTH KOREA', 'LOW'),
    'TW': ('TAIWAN', 'ELEVATED'),
    'IN': ('INDIA', 'MODERATE'),
    'VN': ('VIETNAM', 'MODERATE'),
    'TH': ('THAILAND', 'MODERATE'),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Federal Register API Fetch
# MAGIC
# MAGIC Queries the Federal Register API for trade/tariff-related documents from
# MAGIC relevant agencies, classifies them into event types, and maps them to the
# MAGIC trade tariff event schema.

# COMMAND ----------

# -- Country extraction map (name -> ISO-2 code) --
_COUNTRY_NAME_TO_CODE = {
    "china": "CN", "chinese": "CN", "prc": "CN",
    "russia": "RU", "russian": "RU",
    "mexico": "MX", "mexican": "MX",
    "germany": "DE", "german": "DE",
    "japan": "JP", "japanese": "JP",
    "korea": "KR", "korean": "KR", "south korea": "KR",
    "taiwan": "TW", "taiwanese": "TW",
    "india": "IN", "indian": "IN",
    "vietnam": "VN", "vietnamese": "VN",
    "thailand": "TH", "thai": "TH",
    "canada": "CA", "canadian": "CA",
    "iran": "IR", "iranian": "IR",
    "north korea": "KP",
    "syria": "SY", "syrian": "SY",
    "cuba": "CU", "cuban": "CU",
    "venezuela": "VE",
    "turkey": "TR", "turkish": "TR",
    "brazil": "BR", "brazilian": "BR",
    "european union": "EU", "eu": "EU",
    "united kingdom": "GB", "uk": "GB", "british": "GB",
    "france": "FR", "french": "FR",
    "italy": "IT", "italian": "IT",
    "australia": "AU", "australian": "AU",
    "israel": "IL", "israeli": "IL",
    "saudi arabia": "SA",
    "united arab emirates": "AE", "uae": "AE",
}

def _extract_country(text: str) -> tuple:
    """Extract country code and name from text. Returns (code, name) or (None, None)."""
    if not text:
        return None, None
    text_lower = text.lower()
    for name, code in _COUNTRY_NAME_TO_CODE.items():
        if name in text_lower:
            country_info = TRADE_RISK_COUNTRIES.get(code)
            country_name = country_info[0] if country_info else name.upper()
            return code, country_name
    return None, None


def _extract_hts_code(text: str) -> str:
    """Try to extract an HTS code from text; return '0000' if not found."""
    if not text:
        return "0000"
    # Look for 4-digit HTS-like codes
    matches = re.findall(r'\b(\d{4})\b', text)
    for m in matches:
        if m in DEFENSE_HTS_CODES:
            return m
    return "0000"


def _classify_event_type(doc_type: str, title: str) -> str:
    """Map Federal Register document to a trade event type based on title keywords."""
    title_lower = (title or "").lower()

    rules = [
        (["tariff increase", "raise tariff", "additional duties", "increase in duties"], "TARIFF_INCREASE"),
        (["tariff reduction", "decrease tariff", "lower tariff", "duty reduction"], "TARIFF_DECREASE"),
        (["new tariff", "impose tariff", "imposing duties", "imposition of duties"], "NEW_TARIFF"),
        (["exclusion", "exemption", "waiver", "duty-free"], "TARIFF_EXEMPTION"),
        (["restriction", "restricted"], "TRADE_RESTRICTION"),
        (["export control", "ear ", "export administration", "controlled items"], "EXPORT_CONTROL"),
        (["ban", "prohibit", "embargo"], "IMPORT_BAN"),
        (["agreement", "free trade", "fta "], "TRADE_AGREEMENT"),
        (["sanction", "ofac", "specially designated"], "SANCTION"),
    ]

    for keywords, etype in rules:
        for kw in keywords:
            if kw in title_lower:
                return etype

    return "TRADE_NOTICE"


def _severity_for_doc_type(doc_type: str) -> str:
    """Return severity string based on Federal Register document type."""
    dtype = (doc_type or "").upper()
    if dtype == "RULE":
        return "HIGH"
    elif dtype in ("PRORULE", "PROPOSED RULE"):
        return "ELEVATED"
    else:
        return "MODERATE"


def _impact_score_for_doc_type(doc_type: str) -> float:
    """Return a base impact score with ±10 random variation."""
    dtype = (doc_type or "").upper()
    if dtype == "RULE":
        base = 70
    elif dtype in ("PRORULE", "PROPOSED RULE"):
        base = 50
    else:
        base = 30
    return round(base + random.uniform(-10, 10), 2)

# COMMAND ----------

def fetch_federal_register_trade_events(start_year: int = 2020) -> pd.DataFrame:
    """
    Fetch trade/tariff-related documents from the Federal Register API.

    Runs separate queries for each search term, deduplicates by document_number,
    and maps results to the trade tariff event schema.
    """
    base_url = "https://www.federalregister.gov/api/v1/documents.json"

    search_terms = [
        "tariff",
        "trade restriction",
        "export control",
        "sanctions",
        "import duty",
    ]

    agencies = [
        "international-trade-commission",
        "commerce-department",
        "customs-and-border-protection",
        "international-trade-administration",
        "office-of-the-united-states-trade-representative",
    ]

    fields = [
        "document_number",
        "title",
        "abstract",
        "publication_date",
        "agencies",
        "type",
        "action",
        "dates",
        "docket_ids",
        "regulation_id_numbers",
    ]

    seen_doc_numbers = set()
    all_records = []

    for term in search_terms:
        print(f"Searching Federal Register for '{term}' ...")
        page = 1
        term_count = 0

        while True:
            params = {
                "conditions[term]": term,
                "conditions[publication_date][gte]": f"{start_year}-01-01",
                "per_page": 1000,
                "page": page,
            }
            # Add agency filters
            for agency in agencies:
                params.setdefault("conditions[agencies][]", [])
            # requests handles list params with repeated keys via a list of tuples
            param_tuples = []
            for k, v in params.items():
                if k == "conditions[agencies][]":
                    continue
                param_tuples.append((k, v))
            for agency in agencies:
                param_tuples.append(("conditions[agencies][]", agency))
            for field in fields:
                param_tuples.append(("fields[]", field))

            # Retry logic — 3 attempts with backoff
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(base_url, params=param_tuples, timeout=30)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        print(f"  Rate limited, waiting {wait}s ...")
                        time.sleep(wait)
                    else:
                        print(f"  HTTP {response.status_code}: {response.text[:200]}")
                        time.sleep(2 ** attempt)
                except requests.exceptions.RequestException as exc:
                    print(f"  Request error (attempt {attempt+1}/3): {exc}")
                    time.sleep(2 ** attempt)

            if response is None or response.status_code != 200:
                print(f"  Stopping pagination for '{term}' after retries")
                break

            data = response.json()
            results = data.get("results", [])
            total_pages = data.get("total_pages", 1)

            if not results:
                break

            for doc in results:
                doc_number = doc.get("document_number")
                if not doc_number or doc_number in seen_doc_numbers:
                    continue
                seen_doc_numbers.add(doc_number)

                title = doc.get("title", "")
                abstract = doc.get("abstract", "")
                combined_text = f"{title} {abstract}"
                doc_type = doc.get("type", "NOTICE")
                pub_date = doc.get("publication_date")

                # Agency name
                doc_agencies = doc.get("agencies", [])
                source = doc_agencies[0].get("name", "UNKNOWN") if doc_agencies else "UNKNOWN"

                # Classify event
                event_type = _classify_event_type(doc_type, title)

                # Extract country
                country_code, country_name = _extract_country(combined_text)

                # Extract HTS code
                hts_code = _extract_hts_code(combined_text)
                hts_description = DEFENSE_HTS_CODES.get(hts_code, "")

                # Severity / impact
                severity = _severity_for_doc_type(doc_type)
                impact_score = _impact_score_for_doc_type(doc_type)

                # Build description
                description = abstract if abstract else title

                all_records.append({
                    "event_id": f"FR-{doc_number}",
                    "event_date": pub_date,
                    "event_type": event_type,
                    "country_code": country_code,
                    "country_name": country_name,
                    "hts_code": hts_code,
                    "hts_description": hts_description,
                    "previous_tariff_rate": None,
                    "new_tariff_rate": None,
                    "rate_change_pct": 0.0,
                    "effective_date": pub_date,  # best available approximation
                    "expiration_date": None,
                    "severity": severity,
                    "impact_score": impact_score,
                    "affected_value_usd": None,
                    "source": source,
                    "description": description[:2000] if description else None,
                    "policy_reference": doc_number,
                })
                term_count += 1

            print(f"  Page {page}/{total_pages}: fetched {len(results)} docs ({term_count} new unique)")

            if page >= total_pages:
                break
            page += 1
            time.sleep(0.5)  # polite pacing

    print(f"\nTotal unique trade/tariff documents: {len(all_records)}")
    return pd.DataFrame(all_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

# Fetch trade/tariff events from Federal Register API
start_year = int(dbutils.widgets.get("start_year"))
print(f"Fetching Federal Register trade/tariff documents from {start_year} onward ...")
tariff_df = fetch_federal_register_trade_events(start_year=start_year)
print(f"Fetched {len(tariff_df)} trade/tariff event records")

# COMMAND ----------

# Convert to Spark DataFrame
spark_tariff = spark.createDataFrame(tariff_df)

# Display schema
print("Tariff/Trade Event Schema:")
spark_tariff.printSchema()

# COMMAND ----------

# Display sample
display(spark_tariff.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog setup

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
print(f"Catalog {CATALOG} and schemas raw, silver, gold ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Raw Layer

# COMMAND ----------

# Save to raw layer (Unity Catalog)
spark_tariff.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(RAW_TABLE)

print(f"Saved {spark_tariff.count()} records to {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process to Silver Layer

# COMMAND ----------

# Read raw data (Unity Catalog)
tariff_raw = spark.table(RAW_TABLE)

# Process to silver layer
trade_risk_events = tariff_raw \
    .withColumn("event_date", F.to_date(F.col("event_date"))) \
    .withColumn("effective_date", F.to_date(F.col("effective_date"))) \
    .withColumn("event_month", F.date_trunc("month", F.col("event_date"))) \
    .withColumn("is_negative_event", 
        F.when(F.col("event_type").isin([
            'TARIFF_INCREASE', 'NEW_TARIFF', 'TRADE_RESTRICTION', 
            'EXPORT_CONTROL', 'IMPORT_BAN', 'SANCTION'
        ]), True).otherwise(False)
    ) \
    .withColumn("product_category",
        F.when(F.col("hts_code").startswith("87"), "VEHICLES")
         .when(F.col("hts_code").startswith("73"), "STEEL_STRUCTURES")
         .when(F.col("hts_code").startswith("85"), "ELECTRONICS")
         .when(F.col("hts_code").startswith("72"), "IRON_STEEL")
         .when(F.col("hts_code").startswith("76"), "ALUMINUM")
         .when(F.col("hts_code").startswith("74"), "COPPER")
         .when(F.col("hts_code").startswith("40"), "RUBBER_TIRES")
         .when(F.col("hts_code").startswith("93"), "ARMS_AMMUNITION")
         .otherwise("OTHER")
    ) \
    .withColumn("ingestion_timestamp", F.current_timestamp())

# COMMAND ----------

# Display processed data
display(trade_risk_events.limit(15))

# COMMAND ----------

# Save to silver layer (Unity Catalog)
trade_risk_events.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

print(f"Saved {trade_risk_events.count()} records to {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Gold Layer - Monthly Trade Risk Scores

# COMMAND ----------

# Aggregate to monthly trade risk scores
monthly_trade_risk = trade_risk_events.groupBy("event_month").agg(
    F.count("*").alias("event_count"),
    F.sum(F.when(F.col("is_negative_event"), 1).otherwise(0)).alias("negative_event_count"),
    F.avg("impact_score").alias("avg_impact_score"),
    F.max("impact_score").alias("max_impact_score"),
    F.sum("affected_value_usd").alias("total_affected_value_usd"),
    F.avg("rate_change_pct").alias("avg_rate_change_pct"),
    F.sum(F.when(F.col("severity") == "CRITICAL", 1).otherwise(0)).alias("critical_events"),
    F.sum(F.when(F.col("severity") == "HIGH", 1).otherwise(0)).alias("high_events"),
    F.countDistinct("country_code").alias("countries_affected"),
    F.countDistinct("hts_code").alias("product_categories_affected")
)

# Calculate tariff risk index
window_spec = Window.orderBy("event_count")

trade_risk_monthly = monthly_trade_risk \
    .withColumn("event_score", F.percent_rank().over(window_spec) * 25) \
    .withColumn("severity_score", 
        (F.col("critical_events") * 15 + F.col("high_events") * 8) / 
        F.greatest(F.col("event_count"), F.lit(1)) * 35
    ) \
    .withColumn("impact_score_component", 
        F.col("avg_impact_score") / 100 * 25
    ) \
    .withColumn("breadth_score",
        (F.col("countries_affected") + F.col("product_categories_affected")) / 20 * 15
    ) \
    .withColumn("tariff_risk_index", 
        F.least(
            F.col("event_score") + F.col("severity_score") + 
            F.col("impact_score_component") + F.col("breadth_score"),
            F.lit(100)
        )
    ) \
    .withColumn("risk_level",
        F.when(F.col("tariff_risk_index") >= 70, "CRITICAL")
         .when(F.col("tariff_risk_index") >= 50, "HIGH")
         .when(F.col("tariff_risk_index") >= 30, "ELEVATED")
         .otherwise("MODERATE")
    ) \
    .withColumn("month", F.col("event_month")) \
    .select(
        "month",
        "event_count",
        "negative_event_count",
        "avg_impact_score",
        "max_impact_score",
        "total_affected_value_usd",
        "avg_rate_change_pct",
        "critical_events",
        "high_events",
        "countries_affected",
        "product_categories_affected",
        "tariff_risk_index",
        "risk_level"
    ) \
    .withColumn("ingestion_timestamp", F.current_timestamp())

# COMMAND ----------

# Display monthly trade risk
display(trade_risk_monthly.orderBy(F.desc("month")).limit(24))

# COMMAND ----------

# Save to gold layer (Unity Catalog)
trade_risk_monthly.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(GOLD_TABLE)

print(f"Saved {trade_risk_monthly.count()} monthly trade risk scores to {GOLD_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trade Risk Analysis

# COMMAND ----------

# Events by country
print("=== Trade Events by Country ===")
display(trade_risk_events.groupBy("country_code", "country_name").agg(
    F.count("*").alias("event_count"),
    F.sum(F.when(F.col("is_negative_event"), 1).otherwise(0)).alias("negative_events"),
    F.avg("impact_score").alias("avg_impact")
).orderBy(F.desc("event_count")).limit(15))

# COMMAND ----------

# Events by product category
print("\n=== Trade Events by Product Category ===")
display(trade_risk_events.groupBy("product_category").agg(
    F.count("*").alias("event_count"),
    F.sum("affected_value_usd").alias("total_affected_value"),
    F.avg("rate_change_pct").alias("avg_rate_change")
).orderBy(F.desc("total_affected_value")))

# COMMAND ----------

# High-impact events
print("\n=== Recent High-Impact Trade Events ===")
display(trade_risk_events \
    .filter(F.col("severity").isin(["CRITICAL", "HIGH"])) \
    .select("event_date", "event_type", "country_name", "hts_description", "severity", "impact_score") \
    .orderBy(F.desc("event_date")) \
    .limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables are in Unity Catalog: `supply_chain.raw.trade_tariff_events`, `supply_chain.silver.trade_tariff_risk_events`, `supply_chain.gold.trade_tariff_risk_monthly`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `07_commodity_ingestion` for commodity price data
# MAGIC 2. Run `08_weather_ingestion` for weather risk data
# MAGIC 3. Proceed to transformation notebooks for unified demand signals
