# Databricks notebook source
# MAGIC %md
# MAGIC # FPDS Contract Data Ingestion (USAspending API)
# MAGIC
# MAGIC **Executive summary:** Loads federal contract data (FPDS) from the USAspending API into raw and bronze layers. Fetches real DoD contract awards filtered by tactical vehicle Product/Service Codes (PSC 23xx, 25xx, 29xx) for supply chain analysis.
# MAGIC
# MAGIC **Data Source**: USAspending API — `POST /api/v2/search/spending_by_award/`
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.fpds_contracts` - Raw contract data from USAspending
# MAGIC - `supply_chain.bronze.fpds_contracts` - Cleaned and standardized contracts

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
RAW_TABLE = f"{CATALOG}.raw.fpds_contracts"
BRONZE_TABLE = f"{CATALOG}.bronze.fpds_contracts"

dbutils.widgets.text("max_pages", "30", "Max API pages")

# FPDS data fields of interest
FPDS_COLUMNS = [
    'contractID',
    'referenceIDV',
    'modNumber',
    'transactionNumber',
    'signedDate',
    'effectiveDate',
    'currentCompletionDate',
    'ultimateCompletionDate',
    'obligatedAmount',
    'baseAndAllOptionsValue',
    'baseAndExercisedOptionsValue',
    'contractingOfficeAgencyID',
    'contractingOfficeID',
    'fundingRequestingAgencyID',
    'vendorName',
    'vendorDUNSNumber',
    'vendorUEI',
    'vendorAddressCity',
    'vendorAddressState',
    'vendorAddressCountry',
    'vendorAddressZIP',
    'placeOfPerformanceCity',
    'placeOfPerformanceState',
    'placeOfPerformanceCountry',
    'productOrServiceCode',
    'principalNAICSCode',
    'descriptionOfContractRequirement',
    'typeOfContractPricing',
    'contractActionType',
    'reasonForModification',
    'numberOfOffersReceived',
    'extentCompeted',
    'solicitationID',
    'lastModifiedDate'
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch FPDS Contract Data from USAspending API
# MAGIC
# MAGIC Queries the USAspending `/api/v2/search/spending_by_award/` endpoint for DoD contracts
# MAGIC filtered by tactical vehicle Product/Service Codes (PSC 23xx, 25xx, 29xx).

# COMMAND ----------

def fetch_fpds_contracts(max_pages: int = 30) -> pd.DataFrame:
    """
    Fetch DoD contract (FPDS) data from the USAspending API.
    Paginates through results, mapping API fields to FPDS-style camelCase columns.
    """
    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

    all_records = []

    for page in range(1, max_pages + 1):
        payload = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "time_period": [{"start_date": "2010-10-01", "end_date": "2025-12-31"}],
                "agencies": [
                    {"type": "awarding", "tier": "toptier", "name": "Department of Defense"}
                ],
                "psc_codes": {"require": [["23"], ["25"], ["29"]]}
            },
            "fields": [
                "Award ID", "Recipient Name", "Start Date", "End Date",
                "Award Amount", "Total Obligation", "Awarding Agency",
                "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency",
                "Contract Award Type", "Description", "NAICS Code",
                "Product/Service Code", "Place of Performance City Code",
                "Place of Performance State Code", "Place of Performance Country Code",
                "recipient_id", "Recipient DUNS Number", "Last Modified Date",
                "generated_internal_id"
            ],
            "page": page,
            "limit": 100,
            "sort": "Award Amount",
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
            effective_date = r.get("Start Date")
            ultimate_completion = r.get("End Date")

            # Derive fiscal year from effective date
            fiscal_year = None
            if effective_date:
                try:
                    dt = datetime.strptime(effective_date, "%Y-%m-%d")
                    fiscal_year = dt.year + 1 if dt.month >= 10 else dt.year
                except ValueError:
                    pass

            all_records.append({
                "contractID": r.get("Award ID"),
                "referenceIDV": None,
                "modNumber": None,
                "transactionNumber": None,
                "signedDate": effective_date,
                "effectiveDate": effective_date,
                "currentCompletionDate": ultimate_completion,
                "ultimateCompletionDate": ultimate_completion,
                "obligatedAmount": r.get("Total Obligation"),
                "baseAndAllOptionsValue": r.get("Award Amount"),
                "baseAndExercisedOptionsValue": None,
                "contractingOfficeAgencyID": r.get("Awarding Sub Agency"),
                "contractingOfficeID": None,
                "fundingRequestingAgencyID": r.get("Funding Agency"),
                "vendorName": r.get("Recipient Name"),
                "vendorDUNSNumber": r.get("Recipient DUNS Number"),
                "vendorUEI": None,
                "vendorAddressCity": None,
                "vendorAddressState": None,
                "vendorAddressCountry": None,
                "vendorAddressZIP": None,
                "placeOfPerformanceCity": r.get("Place of Performance City Code"),
                "placeOfPerformanceState": r.get("Place of Performance State Code"),
                "placeOfPerformanceCountry": r.get("Place of Performance Country Code"),
                "productOrServiceCode": r.get("Product/Service Code"),
                "principalNAICSCode": r.get("NAICS Code"),
                "descriptionOfContractRequirement": r.get("Description"),
                "typeOfContractPricing": None,
                "contractActionType": r.get("Contract Award Type"),
                "reasonForModification": None,
                "numberOfOffersReceived": None,
                "extentCompeted": None,
                "solicitationID": None,
                "lastModifiedDate": r.get("Last Modified Date"),
                "fiscalYear": fiscal_year,
            })

        print(f"  Page {page}: fetched {len(results)} records (total: {len(all_records)})")

        # Respect rate limits
        if page < max_pages:
            time.sleep(0.5)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=FPDS_COLUMNS + ["fiscalYear"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion
# MAGIC
# MAGIC Fetches real DoD contract data from the USAspending API.

# COMMAND ----------

max_pages = int(dbutils.widgets.get("max_pages"))
print(f"Fetching FPDS contract data from USAspending API (max {max_pages} pages)...")
fpds_df = fetch_fpds_contracts(max_pages=max_pages)
print(f"Fetched {len(fpds_df)} FPDS contract records from USAspending API")

# COMMAND ----------

# Convert to Spark DataFrame
spark_fpds = spark.createDataFrame(fpds_df)

# Display schema
print("FPDS Schema:")
spark_fpds.printSchema()

# COMMAND ----------

# Display sample
display(spark_fpds.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Raw Layer

# COMMAND ----------

# Unity Catalog setup
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")

# Save to raw layer (Unity Catalog)
spark_fpds.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(RAW_TABLE)

print(f"Saved {spark_fpds.count()} records to {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process to Bronze Layer

# COMMAND ----------

# Add metadata and standardize column names
fpds_bronze = spark_fpds \
    .withColumnRenamed("contractID", "contract_id") \
    .withColumnRenamed("referenceIDV", "reference_idv") \
    .withColumnRenamed("modNumber", "mod_number") \
    .withColumnRenamed("transactionNumber", "transaction_number") \
    .withColumnRenamed("signedDate", "signed_date") \
    .withColumnRenamed("effectiveDate", "effective_date") \
    .withColumnRenamed("currentCompletionDate", "current_completion_date") \
    .withColumnRenamed("ultimateCompletionDate", "ultimate_completion_date") \
    .withColumnRenamed("obligatedAmount", "obligated_amount") \
    .withColumnRenamed("baseAndAllOptionsValue", "base_and_all_options_value") \
    .withColumnRenamed("baseAndExercisedOptionsValue", "base_and_exercised_options_value") \
    .withColumnRenamed("contractingOfficeAgencyID", "contracting_office_agency_id") \
    .withColumnRenamed("contractingOfficeID", "contracting_office_id") \
    .withColumnRenamed("fundingRequestingAgencyID", "funding_requesting_agency_id") \
    .withColumnRenamed("vendorName", "vendor_name") \
    .withColumnRenamed("vendorDUNSNumber", "vendor_duns_number") \
    .withColumnRenamed("vendorUEI", "vendor_uei") \
    .withColumnRenamed("vendorAddressCity", "vendor_city") \
    .withColumnRenamed("vendorAddressState", "vendor_state") \
    .withColumnRenamed("vendorAddressCountry", "vendor_country") \
    .withColumnRenamed("vendorAddressZIP", "vendor_zip") \
    .withColumnRenamed("placeOfPerformanceCity", "pop_city") \
    .withColumnRenamed("placeOfPerformanceState", "pop_state") \
    .withColumnRenamed("placeOfPerformanceCountry", "pop_country") \
    .withColumnRenamed("productOrServiceCode", "psc_code") \
    .withColumnRenamed("principalNAICSCode", "naics_code") \
    .withColumnRenamed("descriptionOfContractRequirement", "description") \
    .withColumnRenamed("typeOfContractPricing", "contract_pricing_type") \
    .withColumnRenamed("contractActionType", "action_type") \
    .withColumnRenamed("reasonForModification", "modification_reason") \
    .withColumnRenamed("numberOfOffersReceived", "num_offers") \
    .withColumnRenamed("extentCompeted", "extent_competed") \
    .withColumnRenamed("solicitationID", "solicitation_id") \
    .withColumnRenamed("lastModifiedDate", "last_modified_date") \
    .withColumnRenamed("fiscalYear", "fiscal_year") \
    .withColumn("ingestion_timestamp", F.current_timestamp()) \
    .withColumn("source_system", F.lit("usaspending_api")) \
    .withColumn("data_quality_flag", F.lit("VALID"))

# COMMAND ----------

# Save to bronze layer (Unity Catalog)
fpds_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(BRONZE_TABLE)

print(f"Saved {fpds_bronze.count()} records to {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

# Summary by vendor
print("=== FPDS Contract Summary by Vendor ===")
display(fpds_bronze.groupBy("vendor_name").agg(
    F.count("*").alias("contract_count"),
    F.sum("obligated_amount").alias("total_obligated"),
    F.avg("obligated_amount").alias("avg_obligated")
).orderBy(F.desc("total_obligated")).limit(15))

# COMMAND ----------

# Summary by PSC code
print("\n=== Contracts by Product/Service Code ===")
display(fpds_bronze.groupBy("psc_code").agg(
    F.count("*").alias("contract_count"),
    F.sum("obligated_amount").alias("total_obligated")
).orderBy(F.desc("total_obligated")).limit(10))

# COMMAND ----------

# Summary by fiscal year
print("\n=== Contracts by Fiscal Year ===")
display(fpds_bronze.groupBy("fiscal_year").agg(
    F.count("*").alias("contract_count"),
    F.sum("obligated_amount").alias("total_obligated")
).orderBy("fiscal_year"))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables written to Unity Catalog: `supply_chain.raw.fpds_contracts`, `supply_chain.bronze.fpds_contracts`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `03_subaward_ingestion` for supplier/subaward data
# MAGIC 2. Run `04_sam_entity_ingestion` for supplier geolocation data
# MAGIC 3. Proceed to transformation notebooks
