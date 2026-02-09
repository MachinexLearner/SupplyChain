# Databricks notebook source
# MAGIC %md
# MAGIC # Unified Monthly Demand Signals
# MAGIC
# MAGIC **Executive summary:** Builds the single gold table of monthly demand and risk by joining prime obligations, subawards, geopolitical/trade/commodity/weather risk. Management: this is the main input for DoD metrics and forecasting; run after ingestion notebooks.
# MAGIC
# MAGIC **Depends on:** Bronze/silver/gold tables from ingestion (01–04, 06–08; GDELT removed). Run `00_setup_catalog` and ingestion notebooks first.
# MAGIC
# MAGIC This notebook creates the unified monthly demand signals table by joining:
# MAGIC - Contract/award data (prime obligations)
# MAGIC - Subaward data (supplier spend)
# MAGIC - Trade/tariff risk scores
# MAGIC - Commodity cost pressure
# MAGIC - Weather disruption indices
# MAGIC
# MAGIC **Target Table** (Unity Catalog): `supply_chain.gold.oshkosh_monthly_demand_signals`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
GOLD_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"

# Source tables (Unity Catalog)
BRONZE_AWARDS_TABLE = f"{CATALOG}.bronze.oshkosh_prime_award_actions"
BRONZE_SUBAWARDS_TABLE = f"{CATALOG}.bronze.oshkosh_subawards"
# Geopolitical risk (optional; table from GDELT ingestion—removed; columns filled with 0 if absent)
GOLD_GEO_RISK_TABLE = f"{CATALOG}.gold.geopolitical_risk_scores_monthly"
GOLD_TRADE_RISK_TABLE = f"{CATALOG}.gold.trade_tariff_risk_monthly"
SILVER_COMMODITY_TABLE = f"{CATALOG}.silver.commodity_prices_monthly"
SILVER_WEATHER_TABLE = f"{CATALOG}.silver.weather_risk_monthly"
SILVER_SUPPLIER_GEO_TABLE = f"{CATALOG}.silver.supplier_geolocations"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Source Data

# COMMAND ----------

# Load bronze award data (Unity Catalog)
try:
    awards_df = spark.table(BRONZE_AWARDS_TABLE)
    print(f"Loaded {awards_df.count()} award records")
except Exception as e:
    print(f"Error loading awards: {e}")
    awards_df = None

# COMMAND ----------

# Load bronze subaward data (Unity Catalog)
try:
    subawards_df = spark.table(BRONZE_SUBAWARDS_TABLE)
    print(f"Loaded {subawards_df.count()} subaward records")
except Exception as e:
    print(f"Error loading subawards: {e}")
    subawards_df = None

# COMMAND ----------

# Load geopolitical risk scores (Unity Catalog)
try:
    geo_risk_df = spark.table(GOLD_GEO_RISK_TABLE)
    print(f"Loaded {geo_risk_df.count()} geopolitical risk records")
except Exception as e:
    print(f"Error loading geo risk: {e}")
    geo_risk_df = None

# COMMAND ----------

# Load trade/tariff risk scores (Unity Catalog)
try:
    trade_risk_df = spark.table(GOLD_TRADE_RISK_TABLE)
    print(f"Loaded {trade_risk_df.count()} trade risk records")
except Exception as e:
    print(f"Error loading trade risk: {e}")
    trade_risk_df = None

# COMMAND ----------

# Load commodity prices (Unity Catalog)
try:
    commodity_df = spark.table(SILVER_COMMODITY_TABLE)
    print(f"Loaded {commodity_df.count()} commodity price records")
except Exception as e:
    print(f"Error loading commodities: {e}")
    commodity_df = None

# COMMAND ----------

# Load weather risk (Unity Catalog)
try:
    weather_df = spark.table(SILVER_WEATHER_TABLE)
    print(f"Loaded {weather_df.count()} weather risk records")
except Exception as e:
    print(f"Error loading weather: {e}")
    weather_df = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Prime Contract Obligations by Month

# COMMAND ----------

# Aggregate prime contract obligations
if awards_df is not None:
    prime_monthly = awards_df \
        .withColumn("action_date", F.to_date(F.col("action_date"))) \
        .withColumn("month", F.date_trunc("month", F.col("action_date"))) \
        .groupBy("month") \
        .agg(
            F.sum("federal_action_obligation").alias("prime_obligations_usd"),
            F.count("*").alias("prime_action_count"),
            F.countDistinct("award_id_piid").alias("unique_awards"),
            F.avg("federal_action_obligation").alias("avg_obligation_per_action")
        )
    
    print("Prime contract monthly aggregation:")
    display(prime_monthly.orderBy(F.desc("month")).limit(12))
else:
    prime_monthly = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Subaward Obligations by Month and Subsystem

# COMMAND ----------

# Aggregate subaward obligations
if subawards_df is not None:
    # Total subawards by month
    subaward_monthly = subawards_df \
        .withColumn("action_date", F.to_date(F.col("subaward_action_date"))) \
        .withColumn("month", F.date_trunc("month", F.col("action_date"))) \
        .groupBy("month") \
        .agg(
            F.sum("subaward_amount").alias("subaward_obligations_usd"),
            F.count("*").alias("subaward_count"),
            F.countDistinct("sub_awardee_name").alias("unique_suppliers")
        )
    
    # Subawards by subsystem category
    subaward_by_subsystem = subawards_df \
        .withColumn("action_date", F.to_date(F.col("subaward_action_date"))) \
        .withColumn("month", F.date_trunc("month", F.col("action_date"))) \
        .groupBy("month") \
        .pivot("subsystem_category") \
        .agg(F.sum("subaward_amount"))
    
    # Rename pivoted columns
    for col_name in subaward_by_subsystem.columns:
        if col_name != "month":
            subaward_by_subsystem = subaward_by_subsystem.withColumnRenamed(
                col_name, f"subsystem_{col_name.lower()}_usd"
            )
    
    print("Subaward monthly aggregation:")
    display(subaward_monthly.orderBy(F.desc("month")).limit(12))
else:
    subaward_monthly = None
    subaward_by_subsystem = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Geopolitical Risk by Month

# COMMAND ----------

# Aggregate geopolitical risk across all regions
if geo_risk_df is not None:
    geo_risk_monthly = geo_risk_df \
        .groupBy("month") \
        .agg(
            F.avg("composite_risk_index").alias("geo_risk_index"),
            F.max("composite_risk_index").alias("geo_risk_max"),
            F.sum("event_count").alias("geo_event_count"),
            F.sum("critical_events").alias("geo_critical_events")
        )
    
    print("Geopolitical risk monthly aggregation:")
    display(geo_risk_monthly.orderBy(F.desc("month")).limit(12))
else:
    geo_risk_monthly = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Trade/Tariff Risk by Month

# COMMAND ----------

# Use trade risk directly (already monthly)
if trade_risk_df is not None:
    trade_risk_monthly = trade_risk_df \
        .select(
            F.col("month"),
            F.col("tariff_risk_index"),
            F.col("event_count").alias("trade_event_count"),
            F.col("critical_events").alias("trade_critical_events"),
            F.col("total_affected_value_usd").alias("trade_affected_value_usd")
        )
    
    print("Trade risk monthly:")
    display(trade_risk_monthly.orderBy(F.desc("month")).limit(12))
else:
    trade_risk_monthly = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Commodity Cost Pressure by Month

# COMMAND ----------

# Aggregate commodity cost pressure
if commodity_df is not None:
    commodity_monthly = commodity_df \
        .withColumn("month_date", F.to_date(F.col("month"))) \
        .withColumn("month", F.date_trunc("month", F.col("month_date"))) \
        .groupBy("month") \
        .agg(
            F.avg("cost_pressure_score").alias("commodity_cost_pressure"),
            F.avg("pct_change_1mo").alias("commodity_avg_1mo_change"),
            F.avg("pct_change_3mo").alias("commodity_avg_3mo_change"),
            F.max(F.abs(F.col("pct_change_1mo"))).alias("commodity_max_volatility")
        )
    
    print("Commodity cost pressure monthly:")
    display(commodity_monthly.orderBy(F.desc("month")).limit(12))
else:
    commodity_monthly = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Weather Disruption by Month

# COMMAND ----------

# Aggregate weather disruption across all locations
if weather_df is not None:
    weather_monthly = weather_df \
        .withColumn("month_date", F.to_date(F.col("month"))) \
        .withColumn("month", F.date_trunc("month", F.col("month_date"))) \
        .groupBy("month") \
        .agg(
            F.avg("weather_disruption_index").alias("weather_disruption_index"),
            F.max("weather_disruption_index").alias("weather_disruption_max"),
            F.sum("extreme_heat_days").alias("total_extreme_heat_days"),
            F.sum("extreme_cold_days").alias("total_extreme_cold_days"),
            F.sum("storm_event_count").alias("total_storm_events")
        )
    
    print("Weather disruption monthly:")
    display(weather_monthly.orderBy(F.desc("month")).limit(12))
else:
    weather_monthly = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Unified Demand Signals Table

# COMMAND ----------

# Generate complete month range
from pyspark.sql.functions import explode, sequence, to_date, lit

# Get date range from data
min_date = datetime(2010, 1, 1)
max_date = datetime.now()

# Create month spine
months_df = spark.range(1).select(
    explode(
        sequence(
            to_date(lit(min_date.strftime('%Y-%m-%d'))),
            to_date(lit(max_date.strftime('%Y-%m-%d'))),
            F.expr("interval 1 month")
        )
    ).alias("month")
)

print(f"Generated {months_df.count()} months from {min_date} to {max_date}")

# COMMAND ----------

# Join all data sources
unified_signals = months_df

# Join prime contract data
if prime_monthly is not None:
    unified_signals = unified_signals.join(
        prime_monthly, on="month", how="left"
    )

# Join subaward data
if subaward_monthly is not None:
    unified_signals = unified_signals.join(
        subaward_monthly, on="month", how="left"
    )

# Join subaward by subsystem
if subaward_by_subsystem is not None:
    unified_signals = unified_signals.join(
        subaward_by_subsystem, on="month", how="left"
    )

# Join geopolitical risk
if geo_risk_monthly is not None:
    unified_signals = unified_signals.join(
        geo_risk_monthly, on="month", how="left"
    )

# Join trade/tariff risk
if trade_risk_monthly is not None:
    unified_signals = unified_signals.join(
        trade_risk_monthly, on="month", how="left"
    )

# Join commodity cost pressure
if commodity_monthly is not None:
    unified_signals = unified_signals.join(
        commodity_monthly, on="month", how="left"
    )

# Join weather disruption
if weather_monthly is not None:
    unified_signals = unified_signals.join(
        weather_monthly, on="month", how="left"
    )

# Ensure geo risk columns exist when GDELT/geopolitical_risk table is not used (e.g. after removing GDELT ingestion)
for col_name in ["geo_risk_index", "geo_risk_max", "geo_event_count", "geo_critical_events"]:
    if col_name not in unified_signals.columns:
        unified_signals = unified_signals.withColumn(col_name, F.lit(0))

# COMMAND ----------

# Fill nulls with zeros for numeric columns
numeric_cols = [col for col in unified_signals.columns if col != "month"]

for col_name in numeric_cols:
    unified_signals = unified_signals.withColumn(
        col_name,
        F.coalesce(F.col(col_name), F.lit(0))
    )

# COMMAND ----------

# Add derived features
unified_signals = unified_signals \
    .withColumn("total_obligations_usd",
        F.col("prime_obligations_usd") + F.col("subaward_obligations_usd")
    ) \
    .withColumn("subaward_ratio",
        F.when(F.col("prime_obligations_usd") > 0,
               F.col("subaward_obligations_usd") / F.col("prime_obligations_usd"))
         .otherwise(F.lit(0))
    ) \
    .withColumn("combined_risk_index",
        (F.col("geo_risk_index") * 0.4 +
         F.col("tariff_risk_index") * 0.3 +
         F.col("weather_disruption_index") * 0.2 +
         F.abs(F.col("commodity_cost_pressure")) * 0.1)
    ) \
    .withColumn("risk_level",
        F.when(F.col("combined_risk_index") >= 50, "CRITICAL")
         .when(F.col("combined_risk_index") >= 35, "HIGH")
         .when(F.col("combined_risk_index") >= 20, "ELEVATED")
         .otherwise("MODERATE")
    ) \
    .withColumn("fiscal_year",
        F.when(F.month("month") >= 10, F.year("month") + 1)
         .otherwise(F.year("month"))
    ) \
    .withColumn("fiscal_quarter",
        F.when(F.month("month").isin([10, 11, 12]), F.lit("Q1"))
         .when(F.month("month").isin([1, 2, 3]), F.lit("Q2"))
         .when(F.month("month").isin([4, 5, 6]), F.lit("Q3"))
         .otherwise(F.lit("Q4"))
    ) \
    .withColumn("ingestion_timestamp", F.current_timestamp())

# COMMAND ----------

# Display final unified signals
display(unified_signals.orderBy(F.desc("month")).limit(24))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Gold Layer

# COMMAND ----------

# Save to gold layer (Unity Catalog)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
unified_signals.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(GOLD_TABLE)

print(f"Saved {unified_signals.count()} records to {GOLD_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

# Summary statistics
print("=== Unified Demand Signals Summary ===")
print(f"\nTotal months: {unified_signals.count()}")
print(f"Date range: {unified_signals.agg(F.min('month')).collect()[0][0]} to {unified_signals.agg(F.max('month')).collect()[0][0]}")

# COMMAND ----------

# Obligations summary
print("\n=== Obligations Summary ===")
display(unified_signals.agg(
    F.sum("prime_obligations_usd").alias("total_prime_obligations"),
    F.sum("subaward_obligations_usd").alias("total_subaward_obligations"),
    F.avg("prime_obligations_usd").alias("avg_monthly_prime"),
    F.avg("subaward_obligations_usd").alias("avg_monthly_subaward")
))

# COMMAND ----------

# Risk summary
print("\n=== Risk Summary ===")
display(unified_signals.agg(
    F.avg("geo_risk_index").alias("avg_geo_risk"),
    F.avg("tariff_risk_index").alias("avg_tariff_risk"),
    F.avg("weather_disruption_index").alias("avg_weather_risk"),
    F.avg("commodity_cost_pressure").alias("avg_commodity_pressure"),
    F.avg("combined_risk_index").alias("avg_combined_risk")
))

# COMMAND ----------

# By fiscal year
print("\n=== Summary by Fiscal Year ===")
display(unified_signals.groupBy("fiscal_year").agg(
    F.sum("prime_obligations_usd").alias("total_obligations"),
    F.avg("combined_risk_index").alias("avg_risk_index")
).orderBy("fiscal_year"))

# COMMAND ----------

# MAGIC %md
# MAGIC Table is in Unity Catalog: `supply_chain.gold.oshkosh_monthly_demand_signals`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `02_dod_metrics_inputs` to calculate DoD metric inputs
# MAGIC 2. Proceed to forecasting notebooks