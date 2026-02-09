# Databricks notebook source
# MAGIC %md
# MAGIC # Subaward / Supplier Data Ingestion (USAspending API)
# MAGIC
# MAGIC **Executive summary:** Loads subaward (prime-to-supplier) data from the USAspending Subaward API into raw and bronze layers. Fetches real DoD subaward records for supply chain supplier analysis and risk monitoring.
# MAGIC
# MAGIC **Data Source**: USAspending API — `POST /api/v2/search/spending_by_award/` with `subawards: true`
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.usa_spending_subawards` - Raw subaward data from USAspending
# MAGIC - `supply_chain.bronze.oshkosh_subawards` - Filtered Oshkosh-related subawards

# COMMAND ----------

# MAGIC %pip install requests pandas pyarrow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import requests
import pandas as pd
import time
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.usa_spending_subawards"
BRONZE_TABLE = f"{CATALOG}.bronze.oshkosh_subawards"

dbutils.widgets.text("max_pages", "35", "Max API pages")

# Oshkosh name variants (as prime contractor)
OSHKOSH_PRIME_VARIANTS = [
    "OSHKOSH DEFENSE",
    "OSHKOSH CORPORATION",
    "OSHKOSH TRUCK",
    "OSHKOSH DEFENSE LLC"
]

# Key suppliers for defense vehicles (tier 1 and tier 2)
DEFENSE_SUPPLIERS = [
    # Powertrain
    ('CATERPILLAR INC', 'IL', 'PEORIA', 'POWERTRAIN'),
    ('CUMMINS INC', 'IN', 'COLUMBUS', 'POWERTRAIN'),
    ('ALLISON TRANSMISSION', 'IN', 'INDIANAPOLIS', 'POWERTRAIN'),
    ('DETROIT DIESEL CORPORATION', 'MI', 'DETROIT', 'POWERTRAIN'),
    
    # Suspension/Axles
    ('MERITOR INC', 'MI', 'TROY', 'SUSPENSION'),
    ('DANA INCORPORATED', 'OH', 'MAUMEE', 'SUSPENSION'),
    ('HENDRICKSON USA', 'IL', 'WOODRIDGE', 'SUSPENSION'),
    ('TAK-4 SYSTEMS LLC', 'WI', 'OSHKOSH', 'SUSPENSION'),
    
    # Armor/Protection
    ('PLASAN NORTH AMERICA', 'MI', 'WALKER', 'ARMOR'),
    ('ARMORWORKS ENTERPRISES', 'AZ', 'TEMPE', 'ARMOR'),
    ('CERADYNE INC', 'CA', 'COSTA MESA', 'ARMOR'),
    ('HARDWIRE LLC', 'MD', 'POCOMOKE CITY', 'ARMOR'),
    
    # Electronics/C4ISR
    ('L3HARRIS TECHNOLOGIES', 'FL', 'MELBOURNE', 'ELECTRONICS'),
    ('RAYTHEON COMPANY', 'MA', 'WALTHAM', 'ELECTRONICS'),
    ('GENERAL DYNAMICS MISSION SYSTEMS', 'VA', 'FAIRFAX', 'ELECTRONICS'),
    ('BAE SYSTEMS ELECTRONICS', 'NH', 'NASHUA', 'ELECTRONICS'),
    ('HONEYWELL AEROSPACE', 'AZ', 'PHOENIX', 'ELECTRONICS'),
    
    # Tires/Wheels
    ('MICHELIN NORTH AMERICA', 'SC', 'GREENVILLE', 'TIRES'),
    ('GOODYEAR TIRE & RUBBER', 'OH', 'AKRON', 'TIRES'),
    ('BRIDGESTONE AMERICAS', 'TN', 'NASHVILLE', 'TIRES'),
    
    # HVAC/Climate
    ('BERGSTROM INC', 'WI', 'ROCKFORD', 'HVAC'),
    ('MOBILE CLIMATE CONTROL', 'ON', 'TORONTO', 'HVAC'),
    
    # Glass/Windows
    ('PPG INDUSTRIES', 'PA', 'PITTSBURGH', 'GLASS'),
    ('SAINT-GOBAIN SEKURIT', 'OH', 'ROSSFORD', 'GLASS'),
    
    # Seats/Interior
    ('SEATS INC', 'WI', 'REEDSBURG', 'INTERIOR'),
    ('FREEDMAN SEATING', 'IL', 'CHICAGO', 'INTERIOR'),
    
    # Hydraulics
    ('PARKER HANNIFIN', 'OH', 'CLEVELAND', 'HYDRAULICS'),
    ('EATON CORPORATION', 'OH', 'CLEVELAND', 'HYDRAULICS'),
    ('BOSCH REXROTH', 'SC', 'FOUNTAIN INN', 'HYDRAULICS'),
    
    # Electrical Systems
    ('APTIV PLC', 'MI', 'TROY', 'ELECTRICAL'),
    ('YAZAKI NORTH AMERICA', 'MI', 'CANTON', 'ELECTRICAL'),
    ('TE CONNECTIVITY', 'PA', 'BERWYN', 'ELECTRICAL'),
    
    # Steel/Materials
    ('NUCOR CORPORATION', 'NC', 'CHARLOTTE', 'MATERIALS'),
    ('UNITED STATES STEEL', 'PA', 'PITTSBURGH', 'MATERIALS'),
    ('ALCOA CORPORATION', 'PA', 'PITTSBURGH', 'MATERIALS'),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Subaward Data from USAspending API
# MAGIC
# MAGIC Queries the USAspending `/api/v2/search/spending_by_award/` endpoint with `subawards: true`
# MAGIC for DoD contract subawards.

# COMMAND ----------

def fetch_usaspending_subawards(max_pages: int = 35) -> pd.DataFrame:
    """
    Fetch DoD subaward data from the USAspending API.
    Paginates through results, mapping API fields to the existing subaward schema.
    """
    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    all_records = []

    for page in range(1, max_pages + 1):
        payload = {
            "subawards": True,
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "time_period": [{"start_date": "2010-10-01", "end_date": "2025-12-31"}],
                "agencies": [
                    {"type": "awarding", "tier": "toptier", "name": "Department of Defense"}
                ]
            },
            "fields": [
                "Sub-Award ID", "Sub-Awardee Name", "Sub-Award Date",
                "Sub-Award Amount", "Awarding Agency", "Awarding Sub Agency",
                "Funding Agency", "Prime Award ID", "Prime Recipient Name",
                "Sub-Award Type", "Prime Award Amount",
                "Sub-Award Place of Performance City Code",
                "Sub-Award Place of Performance State Code",
                "Sub-Award Place of Performance Country Code",
                "Sub-Award Description"
            ],
            "page": page,
            "limit": 100,
            "sort": "Sub-Award Amount",
            "order": "desc"
        }

        # Retry logic with exponential backoff
        data = {"results": []}
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retry {attempt + 1}/3 for page {page} after {wait}s — {e}")
                    time.sleep(wait)
                else:
                    print(f"  FAILED page {page} after 3 attempts — {e}")

        results = data.get("results", [])
        if not results:
            print(f"  No more results at page {page}. Stopping.")
            break

        for r in results:
            action_date = r.get("Sub-Award Date")

            # Derive fiscal year from subaward action date
            fiscal_year = None
            if action_date:
                try:
                    dt = datetime.strptime(action_date, "%Y-%m-%d")
                    fiscal_year = dt.year + 1 if dt.month >= 10 else dt.year
                except ValueError:
                    pass

            all_records.append({
                "subaward_id": r.get("Sub-Award ID"),
                "prime_award_id": r.get("Prime Award ID"),
                "prime_award_type": None,
                "subaward_type": r.get("Sub-Award Type"),
                "subaward_number": None,
                "subaward_amount": r.get("Sub-Award Amount"),
                "subaward_action_date": action_date,
                "subaward_action_date_fiscal_year": fiscal_year,
                "prime_award_amount": r.get("Prime Award Amount"),
                "prime_awardee_name": r.get("Prime Recipient Name"),
                "prime_awardee_uei": None,
                "prime_awardee_city": None,
                "prime_awardee_state": None,
                "prime_awardee_country": None,
                "prime_awardee_zip": None,
                "sub_awardee_name": r.get("Sub-Awardee Name"),
                "sub_awardee_uei": None,
                "sub_awardee_city": None,
                "sub_awardee_state": None,
                "sub_awardee_country": None,
                "sub_awardee_zip": None,
                "sub_place_of_performance_city": r.get("Sub-Award Place of Performance City Code"),
                "sub_place_of_performance_state": r.get("Sub-Award Place of Performance State Code"),
                "sub_place_of_performance_country": r.get("Sub-Award Place of Performance Country Code"),
                "awarding_agency_name": r.get("Awarding Agency"),
                "awarding_sub_agency_name": r.get("Awarding Sub Agency"),
                "funding_agency_name": r.get("Funding Agency"),
                "naics_code": None,
                "naics_description": None,
                "product_or_service_code": None,
                "subaward_description": r.get("Sub-Award Description"),
                "subsystem_category": None,
                "high_comp_officer1_full_name": None,
                "high_comp_officer1_amount": None,
                "last_modified_date": None,
            })

        print(f"  Page {page}: fetched {len(results)} records (total: {len(all_records)})")

        # Respect rate limits
        if page < max_pages:
            time.sleep(0.5)

    if not all_records:
        return pd.DataFrame(columns=[
            "subaward_id", "prime_award_id", "prime_award_type", "subaward_type",
            "subaward_number", "subaward_amount", "subaward_action_date",
            "subaward_action_date_fiscal_year", "prime_award_amount",
            "prime_awardee_name", "prime_awardee_uei", "prime_awardee_city",
            "prime_awardee_state", "prime_awardee_country", "prime_awardee_zip",
            "sub_awardee_name", "sub_awardee_uei", "sub_awardee_city",
            "sub_awardee_state", "sub_awardee_country", "sub_awardee_zip",
            "sub_place_of_performance_city", "sub_place_of_performance_state",
            "sub_place_of_performance_country", "awarding_agency_name",
            "awarding_sub_agency_name", "funding_agency_name", "naics_code",
            "naics_description", "product_or_service_code", "subaward_description",
            "subsystem_category", "high_comp_officer1_full_name",
            "high_comp_officer1_amount", "last_modified_date"
        ])
    return pd.DataFrame(all_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

max_pages = int(dbutils.widgets.get("max_pages"))
print(f"Fetching subaward data from USAspending API (max {max_pages} pages)...")
subaward_df = fetch_usaspending_subawards(max_pages=max_pages)
print(f"Fetched {len(subaward_df)} subaward records from USAspending API")

# COMMAND ----------

# Convert to Spark DataFrame
spark_subaward = spark.createDataFrame(subaward_df)

# Display schema
print("Subaward Schema:")
spark_subaward.printSchema()

# COMMAND ----------

# Display sample
display(spark_subaward.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Raw Layer

# COMMAND ----------

# Unity Catalog setup
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")

# Save to raw layer (Unity Catalog)
spark_subaward.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(RAW_TABLE)

print(f"Saved {spark_subaward.count()} records to {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter for Oshkosh-Related Subawards - Bronze Layer

# COMMAND ----------

# Filter for Oshkosh as prime contractor
oshkosh_filter = F.col("prime_awardee_name").rlike("|".join([f"(?i){name}" for name in OSHKOSH_PRIME_VARIANTS]))

oshkosh_subawards = spark_subaward.filter(oshkosh_filter)

print(f"Filtered to {oshkosh_subawards.count()} Oshkosh-related subawards")

# COMMAND ----------

# Add metadata columns
oshkosh_subawards_bronze = oshkosh_subawards \
    .withColumn("ingestion_timestamp", F.current_timestamp()) \
    .withColumn("source_system", F.lit("usaspending_api")) \
    .withColumn("data_quality_flag", F.lit("VALID"))

# COMMAND ----------

# Save to bronze layer (Unity Catalog)
oshkosh_subawards_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(BRONZE_TABLE)

print(f"Saved {oshkosh_subawards_bronze.count()} Oshkosh subawards to {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supplier Analysis

# COMMAND ----------

# Top suppliers by subaward amount
print("=== Top Oshkosh Suppliers by Subaward Amount ===")
display(oshkosh_subawards_bronze.groupBy("sub_awardee_name", "subsystem_category").agg(
    F.count("*").alias("subaward_count"),
    F.sum("subaward_amount").alias("total_subaward_amount"),
    F.avg("subaward_amount").alias("avg_subaward_amount")
).orderBy(F.desc("total_subaward_amount")).limit(20))

# COMMAND ----------

# Subawards by subsystem category
print("\n=== Subawards by Subsystem Category ===")
display(oshkosh_subawards_bronze.groupBy("subsystem_category").agg(
    F.count("*").alias("subaward_count"),
    F.sum("subaward_amount").alias("total_amount"),
    F.countDistinct("sub_awardee_name").alias("unique_suppliers")
).orderBy(F.desc("total_amount")))

# COMMAND ----------

# Subawards by fiscal year
print("\n=== Subawards by Fiscal Year ===")
display(oshkosh_subawards_bronze.groupBy("subaward_action_date_fiscal_year").agg(
    F.count("*").alias("subaward_count"),
    F.sum("subaward_amount").alias("total_amount")
).orderBy("subaward_action_date_fiscal_year"))

# COMMAND ----------

# Geographic distribution of suppliers
print("\n=== Supplier Geographic Distribution ===")
display(oshkosh_subawards_bronze.groupBy("sub_awardee_state", "sub_awardee_country").agg(
    F.count("*").alias("subaward_count"),
    F.sum("subaward_amount").alias("total_amount"),
    F.countDistinct("sub_awardee_name").alias("unique_suppliers")
).orderBy(F.desc("total_amount")).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables written to Unity Catalog: `supply_chain.raw.usa_spending_subawards`, `supply_chain.bronze.oshkosh_subawards`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supplier Risk Indicators

# COMMAND ----------

# Calculate supplier concentration risk
supplier_concentration = oshkosh_subawards_bronze.groupBy("subsystem_category").agg(
    F.countDistinct("sub_awardee_name").alias("num_suppliers"),
    F.sum("subaward_amount").alias("total_spend")
).withColumn(
    "concentration_risk",
    F.when(F.col("num_suppliers") == 1, "CRITICAL")
     .when(F.col("num_suppliers") <= 2, "HIGH")
     .when(F.col("num_suppliers") <= 4, "MODERATE")
     .otherwise("LOW")
)

print("=== Supplier Concentration Risk by Subsystem ===")
display(supplier_concentration.orderBy("num_suppliers"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `04_sam_entity_ingestion` for supplier entity and geolocation data
# MAGIC 2. Run `06_tariff_trade_ingestion`, `07_commodity_ingestion`, `08_weather_ingestion` for risk signals
