# Databricks notebook source
# MAGIC %md
# MAGIC # Supply Chain Catalog Setup
# MAGIC
# MAGIC **Executive summary:** Creates the Unity Catalog and schemas used by all supply chain notebooks. Run this **first** before any ingestion or transformation.
# MAGIC
# MAGIC **Run this notebook first** to create the Unity Catalog `supply_chain` and schemas used by all ingestion, transformation, and forecasting notebooks.
# MAGIC
# MAGIC **Requires:**
# MAGIC - An external location (e.g. `supply-chain-databricks`) pointing to your cloud path, or set widget `catalog_location` to your S3/ADLS path.
# MAGIC - Permissions: `CREATE CATALOG`, `CREATE SCHEMA` on the metastore.
# MAGIC
# MAGIC **Creates:**
# MAGIC - Catalog: `supply_chain` (managed storage = widget path)
# MAGIC - Schemas: `raw`, `bronze`, `silver`, `gold`, `models`
# MAGIC
# MAGIC **Production:** Set widget `catalog_location` to the managed path for your environment. See `notebooks/README.md` for deployment checklist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Default: S3 path for catalog managed storage (must be covered by your external location)
dbutils.widgets.text("catalog_location", "s3://supply-chain-databricks/", "Catalog S3 path (e.g. s3://supply-chain-databricks/)")

CATALOG = "supply_chain"
SCHEMAS = ["raw", "bronze", "silver", "gold", "models"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create catalog and schemas

# COMMAND ----------

try:
    _path = dbutils.widgets.get("catalog_location")
except Exception:
    _path = ""
if not _path or str(_path).strip() == "":
    import os
    _path = os.environ.get("UC_CATALOG_LOCATION", "s3://supply-chain-databricks/").strip()
_path = _path.rstrip("/")
print(f"Catalog managed location: {_path}")

# COMMAND ----------

# Create catalog with managed storage (avoids "Public DBFS root is disabled")
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} MANAGED LOCATION '{_path}'")
    print(f"✓ Catalog '{CATALOG}' created or already exists with location: {_path}")
except Exception as e:
    err = str(e)
    print(f"✗ Error creating catalog: {err}")
    if "CREATE CATALOG" in err or "permission" in err.lower():
        print("\n→ Create the catalog in the UI: Data → Catalogs → Create catalog")
        print(f"  Name: {CATALOG}, Manage storage: external location for {_path}")
    raise

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
print(f"✓ Using catalog '{CATALOG}'")

# COMMAND ----------

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"✓ Schema '{CATALOG}.{schema}' created or already exists")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

schemas_df = spark.sql(f"SHOW SCHEMAS IN {CATALOG}")
display(schemas_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ✓ Setup complete. Run ingestion notebooks next in order: `01_usaspending_ingestion` → `02_fpds_ingestion` → `03_subaward_ingestion` → `04_sam_entity_ingestion` → `06_tariff_trade_ingestion` → `07_commodity_ingestion` → `08_weather_ingestion`. Optionally run `09`–`13` for World Bank/IMF/NY Fed/WTO risk indicators.
