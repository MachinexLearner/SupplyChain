# Databricks notebook source
# MAGIC %md
# MAGIC # DoD Metrics Inputs Calculation
# MAGIC
# MAGIC **Executive summary:** Computes DoD supply chain metrics (RO, AAO, safety stock, days of supply, demand volatility, NMCS risk) from unified demand signals. Management: use for compliance reporting and inventory/forecast method recommendations.
# MAGIC
# MAGIC **Depends on:** `supply_chain.gold.oshkosh_monthly_demand_signals` (run `01_unified_demand_signals` first).
# MAGIC
# MAGIC This notebook calculates DoD supply chain metrics based on the DAU (Defense Acquisition University) 
# MAGIC Supply Chain Metrics Guide framework.
# MAGIC
# MAGIC **Reference**: https://dau.edu/tools/dod-supply-chain-metrics-guide
# MAGIC
# MAGIC **Key Metrics**:
# MAGIC - Requirements Objective (RO)
# MAGIC - Approved Acquisition Objective (AAO)
# MAGIC - Safety Stock
# MAGIC - Days of Supply
# MAGIC - Demand Variability
# MAGIC
# MAGIC **Target Table** (Unity Catalog): `supply_chain.gold.dod_metrics_inputs_monthly`

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
GOLD_TABLE = f"{CATALOG}.gold.dod_metrics_inputs_monthly"
DEMAND_SIGNALS_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"

# DoD metric parameters (based on DAU guidelines)
DOD_PARAMS = {
    # Lead time assumptions (days)
    'procurement_lead_time_days': 90,       # Average procurement lead time
    'admin_lead_time_days': 30,             # Administrative processing time
    'production_lead_time_days': 120,       # Manufacturing lead time
    
    # Safety stock parameters
    'service_level_target': 0.95,           # 95% service level
    'safety_stock_factor': 1.65,            # Z-score for 95% service level
    
    # Planning horizons
    'operating_level_days': 30,             # Operating level in days
    'order_ship_time_days': 14,             # Order and ship time
    'reorder_cycle_days': 90,               # Reorder cycle
    
    # Forecast horizon for AAO
    'aao_forecast_years': 2,                # 2-year forecast horizon for AAO
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Demand Signals Data

# COMMAND ----------

# Load unified demand signals (Unity Catalog)
try:
    demand_df = spark.table(DEMAND_SIGNALS_TABLE)
    print(f"Loaded {demand_df.count()} demand signal records")
except Exception as e:
    print(f"Error loading demand signals: {e}")
    # Create synthetic data if not available
    demand_df = None

# COMMAND ----------

# Display sample
if demand_df is not None:
    display(demand_df.orderBy(F.desc("month")).limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Demand Statistics

# COMMAND ----------

if demand_df is not None:
    # Calculate rolling statistics
    window_3mo = Window.orderBy("month").rowsBetween(-2, 0)
    window_6mo = Window.orderBy("month").rowsBetween(-5, 0)
    window_12mo = Window.orderBy("month").rowsBetween(-11, 0)
    
    demand_stats = demand_df \
        .withColumn("demand_proxy_usd", F.col("total_obligations_usd")) \
        .withColumn("days_in_month", F.dayofmonth(F.last_day(F.col("month")))) \
        .withColumn("avg_daily_demand_proxy",
            F.col("demand_proxy_usd") / F.col("days_in_month")
        ) \
        .withColumn("demand_3mo_avg",
            F.avg("demand_proxy_usd").over(window_3mo)
        ) \
        .withColumn("demand_6mo_avg",
            F.avg("demand_proxy_usd").over(window_6mo)
        ) \
        .withColumn("demand_12mo_avg",
            F.avg("demand_proxy_usd").over(window_12mo)
        ) \
        .withColumn("demand_stddev_3mo",
            F.stddev("demand_proxy_usd").over(window_3mo)
        ) \
        .withColumn("demand_stddev_6mo",
            F.stddev("demand_proxy_usd").over(window_6mo)
        ) \
        .withColumn("demand_stddev_12mo",
            F.stddev("demand_proxy_usd").over(window_12mo)
        ) \
        .withColumn("demand_stddev_proxy",
            F.coalesce(F.col("demand_stddev_6mo"), F.col("demand_stddev_3mo"), F.lit(0))
        ) \
        .withColumn("coefficient_of_variation",
            F.when(F.col("demand_6mo_avg") > 0,
                   F.col("demand_stddev_6mo") / F.col("demand_6mo_avg"))
             .otherwise(F.lit(0))
        )
    
    print("Demand statistics calculated")
    display(demand_stats.select(
        "month", "demand_proxy_usd", "avg_daily_demand_proxy",
        "demand_6mo_avg", "demand_stddev_proxy", "coefficient_of_variation"
    ).orderBy(F.desc("month")).limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate DoD Metrics
# MAGIC
# MAGIC ### Requirements Objective (RO)
# MAGIC Per DoDM 4140.01 Vol 10:
# MAGIC > RO = Operating Requirements + Acquisition Lead Time Quantity + Economic Order Quantity
# MAGIC
# MAGIC ### Approved Acquisition Objective (AAO)
# MAGIC > AAO = Requirements Objective + 2 Years Forecasted Demand

# COMMAND ----------

if demand_df is not None:
    # Calculate DoD metrics
    dod_metrics = demand_stats \
        .withColumn("lead_time_days_assumed",
            F.lit(DOD_PARAMS['procurement_lead_time_days'] + 
                  DOD_PARAMS['admin_lead_time_days'])
        ) \
        .withColumn("lead_time_demand",
            # Demand during lead time
            F.col("avg_daily_demand_proxy") * F.col("lead_time_days_assumed")
        ) \
        .withColumn("safety_stock_proxy",
            # Safety Stock = Z * σ * √(Lead Time)
            F.lit(DOD_PARAMS['safety_stock_factor']) * 
            F.col("demand_stddev_proxy") * 
            F.sqrt(F.col("lead_time_days_assumed") / 30)  # Convert to monthly stddev
        ) \
        .withColumn("operating_level_demand",
            # Operating level demand
            F.col("avg_daily_demand_proxy") * F.lit(DOD_PARAMS['operating_level_days'])
        ) \
        .withColumn("reorder_point",
            # Reorder Point = Lead Time Demand + Safety Stock
            F.col("lead_time_demand") + F.col("safety_stock_proxy")
        ) \
        .withColumn("requirements_objective_proxy",
            # RO = Operating Level + Lead Time Demand + Safety Stock
            F.col("operating_level_demand") + 
            F.col("lead_time_demand") + 
            F.col("safety_stock_proxy")
        ) \
        .withColumn("two_year_forecast_demand",
            # 2-year forecast based on 12-month average
            F.col("demand_12mo_avg") * 24
        ) \
        .withColumn("approved_acquisition_objective_proxy",
            # AAO = RO + 2-year forecast
            F.col("requirements_objective_proxy") + F.col("two_year_forecast_demand")
        ) \
        .withColumn("days_of_supply_proxy",
            # Days of Supply = On-hand / Average Daily Demand
            # Using RO as proxy for on-hand
            F.when(F.col("avg_daily_demand_proxy") > 0,
                   F.col("requirements_objective_proxy") / F.col("avg_daily_demand_proxy"))
             .otherwise(F.lit(0))
        ) \
        .withColumn("inventory_turnover_proxy",
            # Turnover = Annual Demand / Average Inventory
            F.when(F.col("requirements_objective_proxy") > 0,
                   (F.col("demand_12mo_avg") * 12) / F.col("requirements_objective_proxy"))
             .otherwise(F.lit(0))
        )
    
    print("DoD metrics calculated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Risk-Adjusted Metrics

# COMMAND ----------

if demand_df is not None:
    # Add risk adjustments
    dod_metrics_final = dod_metrics \
        .withColumn("risk_adjusted_safety_stock",
            # Increase safety stock based on combined risk
            F.col("safety_stock_proxy") * (1 + F.col("combined_risk_index") / 100)
        ) \
        .withColumn("risk_adjusted_ro",
            # Risk-adjusted Requirements Objective
            F.col("operating_level_demand") + 
            F.col("lead_time_demand") + 
            F.col("risk_adjusted_safety_stock")
        ) \
        .withColumn("demand_volatility_category",
            # Categorize demand volatility (per DoD intermittent demand guidelines)
            F.when(F.col("coefficient_of_variation") > 1.0, "HIGHLY_INTERMITTENT")
             .when(F.col("coefficient_of_variation") > 0.5, "INTERMITTENT")
             .when(F.col("coefficient_of_variation") > 0.3, "VARIABLE")
             .otherwise("STABLE")
        ) \
        .withColumn("forecast_method_recommendation",
            # Recommend forecast method based on demand pattern
            F.when(F.col("demand_volatility_category") == "HIGHLY_INTERMITTENT", "CROSTON")
             .when(F.col("demand_volatility_category") == "INTERMITTENT", "SBA")
             .when(F.col("demand_volatility_category") == "VARIABLE", "PROPHET")
             .otherwise("ARIMA")
        ) \
        .withColumn("nmcs_risk_indicator",
            # Not Mission Capable - Supply risk indicator
            F.when(
                (F.col("days_of_supply_proxy") < 30) & (F.col("combined_risk_index") > 40),
                "HIGH_RISK"
            ).when(
                (F.col("days_of_supply_proxy") < 60) | (F.col("combined_risk_index") > 30),
                "ELEVATED_RISK"
            ).otherwise("NORMAL")
        )
    
    print("Risk-adjusted metrics calculated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Final Columns

# COMMAND ----------

if demand_df is not None:
    # Select final columns for gold table
    final_metrics = dod_metrics_final.select(
        # Time dimensions
        "month",
        "fiscal_year",
        "fiscal_quarter",
        
        # Demand metrics
        "demand_proxy_usd",
        "avg_daily_demand_proxy",
        "demand_6mo_avg",
        "demand_12mo_avg",
        "demand_stddev_proxy",
        "coefficient_of_variation",
        "demand_volatility_category",
        
        # Lead time
        "lead_time_days_assumed",
        "lead_time_demand",
        
        # Safety stock
        "safety_stock_proxy",
        "risk_adjusted_safety_stock",
        
        # DoD metrics
        "requirements_objective_proxy",
        "risk_adjusted_ro",
        "approved_acquisition_objective_proxy",
        "days_of_supply_proxy",
        "inventory_turnover_proxy",
        "reorder_point",
        
        # Risk indicators
        "combined_risk_index",
        "risk_level",
        "nmcs_risk_indicator",
        
        # Recommendations
        "forecast_method_recommendation",
        
        # Metadata
        F.current_timestamp().alias("calculation_timestamp")
    )
    
    display(final_metrics.orderBy(F.desc("month")).limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Gold Layer

# COMMAND ----------

if demand_df is not None:
    # Save to gold layer (Unity Catalog)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
    final_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(GOLD_TABLE)
    
    print(f"Saved {final_metrics.count()} records to {GOLD_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DoD Metrics Summary

# COMMAND ----------

if demand_df is not None:
    # Summary statistics
    print("=== DoD Metrics Summary ===")
    display(final_metrics.agg(
        F.avg("demand_proxy_usd").alias("avg_monthly_demand"),
        F.avg("avg_daily_demand_proxy").alias("avg_daily_demand"),
        F.avg("demand_stddev_proxy").alias("avg_demand_stddev"),
        F.avg("safety_stock_proxy").alias("avg_safety_stock"),
        F.avg("requirements_objective_proxy").alias("avg_requirements_objective"),
        F.avg("days_of_supply_proxy").alias("avg_days_of_supply")
    ))

# COMMAND ----------

if demand_df is not None:
    # Demand volatility distribution
    print("\n=== Demand Volatility Distribution ===")
    display(final_metrics.groupBy("demand_volatility_category").agg(
        F.count("*").alias("month_count"),
        F.avg("coefficient_of_variation").alias("avg_cv")
    ).orderBy("demand_volatility_category"))

# COMMAND ----------

if demand_df is not None:
    # NMCS risk distribution
    print("\n=== NMCS Risk Distribution ===")
    display(final_metrics.groupBy("nmcs_risk_indicator").agg(
        F.count("*").alias("month_count"),
        F.avg("days_of_supply_proxy").alias("avg_days_of_supply"),
        F.avg("combined_risk_index").alias("avg_risk_index")
    ).orderBy("nmcs_risk_indicator"))

# COMMAND ----------

if demand_df is not None:
    # Forecast method recommendations
    print("\n=== Forecast Method Recommendations ===")
    display(final_metrics.groupBy("forecast_method_recommendation").agg(
        F.count("*").alias("month_count")
    ).orderBy(F.desc("month_count")))

# COMMAND ----------

# MAGIC %md
# MAGIC Table is in Unity Catalog: `supply_chain.gold.dod_metrics_inputs_monthly`

# COMMAND ----------

# MAGIC %md
# MAGIC ## DoD Metrics Definitions Reference
# MAGIC
# MAGIC | Metric | Definition | Formula |
# MAGIC |--------|------------|---------|
# MAGIC | **Requirements Objective (RO)** | Total quantity needed to support operations | Operating Level + Lead Time Demand + Safety Stock |
# MAGIC | **Approved Acquisition Objective (AAO)** | Maximum quantity authorized for acquisition | RO + 2-Year Forecast Demand |
# MAGIC | **Safety Stock** | Buffer inventory for demand variability | Z × σ × √(Lead Time) |
# MAGIC | **Days of Supply** | How long current inventory will last | On-Hand Inventory / Average Daily Demand |
# MAGIC | **Coefficient of Variation** | Measure of demand variability | Standard Deviation / Mean |
# MAGIC | **NMCS Risk** | Risk of Not Mission Capable - Supply | Based on Days of Supply and Risk Index |
# MAGIC
# MAGIC **Reference**: DAU DoD Supply Chain Metrics Guide (dau.edu/tools/dod-supply-chain-metrics-guide)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Proceed to forecasting notebooks
# MAGIC 2. Run `01_prophet_forecasting` for demand forecasting
# MAGIC 3. Run `02_arima_forecasting` for alternative models