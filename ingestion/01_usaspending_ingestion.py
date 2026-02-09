# Databricks notebook source
# MAGIC %md
# MAGIC # USAspending API Ingestion
# MAGIC
# MAGIC **Executive summary:** Loads federal contract award data from the USAspending.gov Award Search API into raw and bronze tables. Fetches real defense vehicle contracts (PSC codes 23xx, 25xx, 29xx) via paginated API calls.
# MAGIC
# MAGIC **Data Source**: https://api.usaspending.gov/api/v2/search/spending_by_award/
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.usa_spending_awards` - Raw award data
# MAGIC - `supply_chain.bronze.oshkosh_prime_award_actions` - Filtered Oshkosh Defense contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Install required packages
%pip install requests pandas pyarrow

# COMMAND ----------

import requests
import pandas as pd
import os
import time
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Optional widget: set catalog cloud path when "Public DBFS root is disabled"
dbutils.widgets.text("catalog_location", "s3://supply-chain-databricks/", "UC catalog S3 path")
dbutils.widgets.text("max_pages", "50", "Max API pages to fetch (100 records/page)")

# Configuration - Unity Catalog (catalog.schema.table)
CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.usa_spending_awards"
BRONZE_TABLE = f"{CATALOG}.bronze.oshkosh_prime_award_actions"

USASPENDING_API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

OSHKOSH_NAME_VARIANTS = [
    "OSHKOSH DEFENSE",
    "OSHKOSH CORPORATION",
    "OSHKOSH TRUCK",
    "OSHKOSH DEFENSE LLC",
    "OSHKOSH DEFENSE, LLC"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog setup (run `00_setup_catalog` first)

# COMMAND ----------

try:
    _catalog_loc = dbutils.widgets.get("catalog_location")
except Exception:
    _catalog_loc = None
if _catalog_loc is None or str(_catalog_loc).strip() == "":
    _catalog_loc = os.environ.get("UC_CATALOG_LOCATION", "s3://supply-chain-databricks/").strip()

try:
    catalogs_df = spark.sql("SHOW CATALOGS")
    existing_catalogs = [row.catalog for row in catalogs_df.collect()]
    catalog_exists = CATALOG in existing_catalogs
except Exception as e:
    catalog_exists = False

if not catalog_exists and _catalog_loc:
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} MANAGED LOCATION '{_catalog_loc.rstrip('/')}'")
        print(f"✓ Catalog {CATALOG} created with location: {_catalog_loc}")
    except Exception as e:
        print(f"✗ Error creating catalog: {e}")
        raise
else:
    print(f"✓ Catalog {CATALOG} already exists")

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")
print(f"✓ Using catalog {CATALOG}, schemas raw/bronze ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Function – USAspending Award Search API

# COMMAND ----------

def fetch_usaspending_awards(max_pages: int = 50) -> pd.DataFrame:
    """
    Fetch defense vehicle contract awards from the USAspending Award Search API.

    Uses PSC code prefixes 23 (vehicles), 25 (vehicle components), 29 (engine parts)
    to pull all contract types (A/B/C/D) across FY2010-FY2025.

    Args:
        max_pages: Maximum number of pages to fetch (100 records per page).

    Returns:
        pandas DataFrame with records mapped to the standard schema.
    """

    # -- Field mapping from API response keys to our schema columns ----------
    field_map = {
        "Award ID": "award_id_piid",
        "Recipient Name": "recipient_name",
        "Start Date": "period_of_performance_start_date",
        "End Date": "period_of_performance_current_end_date",
        "Award Amount": "current_total_value_of_award",
        "Total Obligation": "federal_action_obligation",
        "Awarding Agency": "awarding_agency_name",
        "Awarding Sub Agency": "awarding_sub_agency_name",
        "Funding Agency": "funding_agency_name",
        "Contract Award Type": "award_type",
        "Description": "award_description",
        "NAICS Code": "naics_code",
        "Product/Service Code": "product_or_service_code",
        "Place of Performance State Code": "primary_place_of_performance_state_code",
        "Place of Performance Country Code": "primary_place_of_performance_country_code",
        "Last Modified Date": "last_modified_date",
        "Recipient DUNS Number": "recipient_uei",
    }

    api_fields = [
        "Award ID", "Recipient Name", "Start Date", "End Date",
        "Award Amount", "Total Obligation", "Awarding Agency",
        "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency",
        "Contract Award Type", "Award Type", "Description",
        "NAICS Code", "Product/Service Code",
        "Place of Performance State Code", "Place of Performance Country Code",
        "recipient_id", "Recipient DUNS Number",
        "Last Modified Date", "generated_internal_id"
    ]

    payload_template = {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],
            "time_period": [{"start_date": "2010-10-01", "end_date": "2025-12-31"}],
            "psc_codes": {
                "require": [["23"], ["25"], ["29"]]
            }
        },
        "fields": api_fields,
        "limit": 100,
        "sort": "Award Amount",
        "order": "desc"
    }

    all_records = []
    page_num = 1
    max_retries = 3

    while page_num <= max_pages:
        payload = {**payload_template, "page": page_num}
        response = None

        # -- Retry loop with exponential backoff ----------------------------
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    USASPENDING_API_URL,
                    json=payload,
                    timeout=60,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                break  # success
            except requests.exceptions.RequestException as exc:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  ⚠ Page {page_num} attempt {attempt} failed: {exc}. Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"USAspending API unreachable after {max_retries} attempts on page {page_num}: {exc}"
                    ) from exc

        data = response.json()
        results = data.get("results", [])

        if not results:
            print(f"  No more results at page {page_num}. Stopping pagination.")
            break

        # -- Map each result row to our schema columns ----------------------
        for row in results:
            mapped = {}
            for api_key, col_name in field_map.items():
                mapped[col_name] = row.get(api_key)

            # Derived / default fields
            mapped["action_date"] = mapped.get("period_of_performance_start_date")
            mapped["total_dollars_obligated"] = mapped.get("federal_action_obligation")

            # Fiscal year: month >= 10 ➜ next calendar year's FY
            try:
                ad = mapped.get("action_date")
                if ad:
                    dt = datetime.strptime(str(ad)[:10], "%Y-%m-%d")
                    mapped["fiscal_year"] = dt.year + 1 if dt.month >= 10 else dt.year
                else:
                    mapped["fiscal_year"] = None
            except (ValueError, TypeError):
                mapped["fiscal_year"] = None

            # Fields not returned by this endpoint – set to None
            mapped.setdefault("award_id_fain", None)
            mapped.setdefault("modification_number", None)
            mapped.setdefault("transaction_number", None)
            mapped.setdefault("parent_award_id_piid", None)
            mapped.setdefault("base_and_exercised_options_value", None)
            mapped.setdefault("potential_total_value_of_award", None)
            mapped.setdefault("recipient_parent_name", None)
            mapped.setdefault("recipient_country_code", None)
            mapped.setdefault("recipient_state_code", None)
            mapped.setdefault("recipient_city_name", None)
            mapped.setdefault("recipient_zip_code", None)
            mapped.setdefault("naics_description", None)
            mapped.setdefault("type_of_contract_pricing", None)
            mapped.setdefault("funding_sub_agency_name", row.get("Funding Sub Agency"))

            all_records.append(mapped)

        print(f"  ✓ Page {page_num}: fetched {len(results)} records (total so far: {len(all_records)})")

        # If fewer than `limit` results came back we've reached the last page
        if len(results) < 100:
            break

        page_num += 1

    if not all_records:
        raise RuntimeError("USAspending API returned zero records. Check filters or connectivity.")

    print(f"\n✓ Fetched {len(all_records)} total records from {page_num} page(s).")
    return pd.DataFrame(all_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion
# MAGIC
# MAGIC Fetches real contract award data from the USAspending Award Search API with defense vehicle PSC-code filters.

# COMMAND ----------

try:
    _max_pages = int(dbutils.widgets.get("max_pages"))
except Exception:
    _max_pages = int(os.environ.get("MAX_PAGES", "50"))

print(f"Fetching awards from USAspending API (max {_max_pages} pages, 100 records/page)…")
usa_spending_df = fetch_usaspending_awards(max_pages=_max_pages)
print(f"Total records retrieved: {len(usa_spending_df)}")

# COMMAND ----------

spark_df = spark.createDataFrame(usa_spending_df)
print("Schema:")
spark_df.printSchema()

# COMMAND ----------

display(spark_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Raw Layer

# COMMAND ----------

# DBTITLE 1,Save all award data to raw layer (Unity Catalog)
# Use saveAsTable so data goes to catalog managed storage (s3://supply-chain-databricks/), not DBFS.
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(RAW_TABLE)

print(f"Saved {spark_df.count()} records to {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter for Oshkosh Defense - Bronze Layer

# COMMAND ----------

oshkosh_filter = F.col("recipient_name").rlike("|".join([f"(?i){name}" for name in OSHKOSH_NAME_VARIANTS]))
oshkosh_df = spark_df.filter(oshkosh_filter)
print(f"Filtered to {oshkosh_df.count()} Oshkosh Defense records")

# COMMAND ----------

oshkosh_bronze = oshkosh_df \
    .withColumn("ingestion_timestamp", F.current_timestamp()) \
    .withColumn("source_system", F.lit("usaspending_api")) \
    .withColumn("data_quality_flag", F.lit("VALID"))

# COMMAND ----------

oshkosh_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(BRONZE_TABLE)

print(f"Saved {oshkosh_bronze.count()} Oshkosh records to {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

print("=== Oshkosh Defense Contract Summary ===")
print(f"\nTotal Records: {oshkosh_bronze.count()}")
display(oshkosh_bronze.groupBy("fiscal_year").agg(
    F.count("*").alias("record_count"),
    F.sum("federal_action_obligation").alias("total_obligations"),
    F.avg("federal_action_obligation").alias("avg_obligation")
).orderBy("fiscal_year"))

# COMMAND ----------

display(oshkosh_bronze.groupBy("product_or_service_code").agg(
    F.count("*").alias("record_count"),
    F.sum("federal_action_obligation").alias("total_obligations")
).orderBy(F.desc("total_obligations")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables: `supply_chain.raw.usa_spending_awards`, `supply_chain.bronze.oshkosh_prime_award_actions`
