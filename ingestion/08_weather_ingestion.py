# Databricks notebook source
# MAGIC %md
# MAGIC # Weather Risk Data Ingestion
# MAGIC
# MAGIC **Executive summary:** Loads weather and climate risk for key supply chain locations into silver (monthly disruption index) using real Meteostat data. Management: feeds combined risk and weather scenario tools.
# MAGIC
# MAGIC **Data Sources**:
# MAGIC - Meteostat (free Python library) - real historical weather observations
# MAGIC - NOAA bulk climate data (no key required)
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.silver.weather_risk_monthly` - Monthly weather risk indicators
# MAGIC
# MAGIC **Requirements:** Cluster must have outbound internet access. Install meteostat via `%pip install meteostat`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Install meteostat
%pip install meteostat pandas

# COMMAND ----------

# Meteostat v2 uses lowercase monthly(); v1 used Monthly class
try:
    from meteostat import Point, monthly
    _METEO_MONTHLY_CALLABLE = monthly
    _METEO_IS_V2 = True
except ImportError:
    from meteostat import Point, Monthly
    _METEO_MONTHLY_CALLABLE = lambda point, start, end: Monthly(point, start, end).fetch()
    _METEO_IS_V2 = False
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
SILVER_TABLE = f"{CATALOG}.silver.weather_risk_monthly"

# Key locations for Oshkosh Defense supply chain
# Manufacturing facilities, major suppliers, and logistics hubs
SUPPLY_CHAIN_LOCATIONS = {
    # Oshkosh facilities
    'oshkosh_hq': {
        'name': 'Oshkosh HQ',
        'city': 'Oshkosh',
        'state': 'WI',
        'lat': 44.0247,
        'lon': -88.5426,
        'type': 'MANUFACTURING'
    },
    'appleton': {
        'name': 'Pierce Manufacturing',
        'city': 'Appleton',
        'state': 'WI',
        'lat': 44.2619,
        'lon': -88.4154,
        'type': 'MANUFACTURING'
    },
    
    # Major supplier locations
    'detroit': {
        'name': 'Detroit Metro (Powertrain)',
        'city': 'Detroit',
        'state': 'MI',
        'lat': 42.3314,
        'lon': -83.0458,
        'type': 'SUPPLIER_HUB'
    },
    'indianapolis': {
        'name': 'Indianapolis (Allison)',
        'city': 'Indianapolis',
        'state': 'IN',
        'lat': 39.7684,
        'lon': -86.1581,
        'type': 'SUPPLIER'
    },
    'cleveland': {
        'name': 'Cleveland (Parker/Eaton)',
        'city': 'Cleveland',
        'state': 'OH',
        'lat': 41.4993,
        'lon': -81.6944,
        'type': 'SUPPLIER_HUB'
    },
    'pittsburgh': {
        'name': 'Pittsburgh (Steel/Materials)',
        'city': 'Pittsburgh',
        'state': 'PA',
        'lat': 40.4406,
        'lon': -79.9959,
        'type': 'SUPPLIER_HUB'
    },
    
    # Logistics hubs
    'chicago': {
        'name': 'Chicago Logistics Hub',
        'city': 'Chicago',
        'state': 'IL',
        'lat': 41.8781,
        'lon': -87.6298,
        'type': 'LOGISTICS'
    },
    'houston': {
        'name': 'Houston Port',
        'city': 'Houston',
        'state': 'TX',
        'lat': 29.7604,
        'lon': -95.3698,
        'type': 'PORT'
    },
    'los_angeles': {
        'name': 'Los Angeles Port',
        'city': 'Los Angeles',
        'state': 'CA',
        'lat': 33.9425,
        'lon': -118.4081,
        'type': 'PORT'
    },
    
    # Military depot locations
    'anniston': {
        'name': 'Anniston Army Depot',
        'city': 'Anniston',
        'state': 'AL',
        'lat': 33.6598,
        'lon': -85.8316,
        'type': 'DEPOT'
    },
    'red_river': {
        'name': 'Red River Army Depot',
        'city': 'Texarkana',
        'state': 'TX',
        'lat': 33.4418,
        'lon': -94.0477,
        'type': 'DEPOT'
    },
}

# Weather thresholds for disruption risk
WEATHER_THRESHOLDS = {
    'extreme_heat_temp_c': 35,      # Above 35°C = extreme heat
    'extreme_cold_temp_c': -15,     # Below -15°C = extreme cold
    'heavy_precip_mm': 100,         # Above 100mm monthly = heavy precipitation
    'drought_precip_mm': 10,        # Below 10mm monthly = drought conditions
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Weather Data

# COMMAND ----------

def fetch_weather_data(locations: dict, years_back: int = 5) -> pd.DataFrame:
    """
    Fetch historical weather data from Meteostat.
    
    Args:
        locations: Dictionary of location metadata
        years_back: Number of years of history to fetch
    
    Returns:
        DataFrame with monthly weather data
    """
    end_date = datetime.now()
    start_date = datetime(end_date.year - years_back, 1, 1)
    
    all_data = []
    
    for loc_id, loc_info in locations.items():
        print(f"Fetching weather for {loc_info['name']}...")
        try:
            # Create point
            point = Point(loc_info['lat'], loc_info['lon'])
            
            # Get monthly data (meteostat v2: monthly() returns DataFrame or None; v1: Monthly().fetch())
            data = _METEO_MONTHLY_CALLABLE(point, start_date, end_date)
            if data is None:
                data = pd.DataFrame()
            elif not isinstance(data, pd.DataFrame):
                data = data.fetch() if hasattr(data, 'fetch') else pd.DataFrame(data)
            if data is None:
                data = pd.DataFrame()
            
            if data.empty:
                print(f"  No data available for {loc_info['name']}")
                continue
            
            # v2 uses 'time' column; v1 uses index as date
            if 'time' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                data = data.set_index('time')
            
            for date, row in data.iterrows():
                all_data.append({
                    'month': date.strftime('%Y-%m-%d'),
                    'location_id': loc_id,
                    'location_name': loc_info['name'],
                    'city': loc_info['city'],
                    'state': loc_info['state'],
                    'lat': loc_info['lat'],
                    'lon': loc_info['lon'],
                    'location_type': loc_info['type'],
                    'tavg': row.get('tavg'),      # Average temperature
                    'tmin': row.get('tmin'),      # Minimum temperature
                    'tmax': row.get('tmax'),      # Maximum temperature
                    'prcp': row.get('prcp'),      # Precipitation
                    'snow': row.get('snow'),      # Snowfall
                    'wdir': row.get('wdir'),      # Wind direction
                    'wspd': row.get('wspd'),      # Wind speed
                    'pres': row.get('pres'),      # Pressure
                })
            
            print(f"  Retrieved {len(data)} months of data")
            
        except Exception as e:
            print(f"  Error fetching {loc_info['name']}: {e}")
    
    return pd.DataFrame(all_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

print("Fetching weather data from Meteostat...")
weather_df = fetch_weather_data(SUPPLY_CHAIN_LOCATIONS, years_back=5)
if weather_df.empty:
    raise RuntimeError(
        "Meteostat returned no data. Ensure the cluster has outbound internet access "
        "and meteostat is installed (pip install meteostat). Check that Meteostat's "
        "data servers are reachable."
    )
print(f"Successfully fetched {len(weather_df)} records from Meteostat")

# COMMAND ----------

# Convert to Spark DataFrame
spark_weather = spark.createDataFrame(weather_df)

# Display schema
print("Weather Data Schema:")
spark_weather.printSchema()

# COMMAND ----------

# Display sample
display(spark_weather.filter(F.col("month") >= "2024-01-01").orderBy("month", "location_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Weather Risk Indicators

# COMMAND ----------

# Add weather risk indicators
weather_risk = spark_weather \
    .withColumn("month_date", F.to_date(F.col("month"))) \
    .withColumn("extreme_heat_days",
        F.when(F.col("tmax") > WEATHER_THRESHOLDS['extreme_heat_temp_c'], 
               F.lit(15)).otherwise(F.lit(0))  # Estimate days based on monthly max
    ) \
    .withColumn("extreme_cold_days",
        F.when(F.col("tmin") < WEATHER_THRESHOLDS['extreme_cold_temp_c'], 
               F.lit(15)).otherwise(F.lit(0))
    ) \
    .withColumn("precipitation_anomaly",
        F.when(F.col("prcp") > WEATHER_THRESHOLDS['heavy_precip_mm'], "HEAVY")
         .when(F.col("prcp") < WEATHER_THRESHOLDS['drought_precip_mm'], "DROUGHT")
         .otherwise("NORMAL")
    ) \
    .withColumn("storm_event_count",
        # Estimate based on precipitation and wind
        F.when((F.col("prcp") > 80) & (F.col("wspd") > 15), F.lit(3))
         .when((F.col("prcp") > 50) | (F.col("wspd") > 20), F.lit(1))
         .otherwise(F.lit(0))
    ) \
    .withColumn("snow_disruption_risk",
        F.when(F.col("snow") > 50, "HIGH")
         .when(F.col("snow") > 20, "MODERATE")
         .when(F.col("snow") > 0, "LOW")
         .otherwise("NONE")
    )

# COMMAND ----------

# Calculate weather disruption index (0-100 scale)
weather_risk_final = weather_risk \
    .withColumn("heat_score",
        F.col("extreme_heat_days") / 30 * 25
    ) \
    .withColumn("cold_score",
        F.col("extreme_cold_days") / 30 * 25
    ) \
    .withColumn("precip_score",
        F.when(F.col("precipitation_anomaly") == "HEAVY", F.lit(20))
         .when(F.col("precipitation_anomaly") == "DROUGHT", F.lit(15))
         .otherwise(F.lit(0))
    ) \
    .withColumn("storm_score",
        F.col("storm_event_count") * 10
    ) \
    .withColumn("snow_score",
        F.when(F.col("snow_disruption_risk") == "HIGH", F.lit(20))
         .when(F.col("snow_disruption_risk") == "MODERATE", F.lit(10))
         .when(F.col("snow_disruption_risk") == "LOW", F.lit(5))
         .otherwise(F.lit(0))
    ) \
    .withColumn("weather_disruption_index",
        F.least(
            F.col("heat_score") + F.col("cold_score") + F.col("precip_score") + 
            F.col("storm_score") + F.col("snow_score"),
            F.lit(100)
        )
    ) \
    .withColumn("disruption_risk_level",
        F.when(F.col("weather_disruption_index") >= 50, "HIGH")
         .when(F.col("weather_disruption_index") >= 25, "MODERATE")
         .otherwise("LOW")
    ) \
    .select(
        "month",
        "month_date",
        "location_id",
        "location_name",
        "city",
        "state",
        "lat",
        "lon",
        "location_type",
        "tavg",
        "tmin",
        "tmax",
        "prcp",
        "snow",
        "wspd",
        "extreme_heat_days",
        "extreme_cold_days",
        "precipitation_anomaly",
        "storm_event_count",
        "snow_disruption_risk",
        "weather_disruption_index",
        "disruption_risk_level"
    ) \
    .withColumn("ingestion_timestamp", F.current_timestamp())

# COMMAND ----------

# Display enriched data
display(weather_risk_final.filter(F.col("month") >= "2024-01-01").orderBy(F.desc("weather_disruption_index")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog setup and Save to Silver Layer

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")
print(f"Catalog {CATALOG} and schema silver ready.")

# COMMAND ----------

# Save to silver layer (Unity Catalog)
weather_risk_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

print(f"Saved {weather_risk_final.count()} records to {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather Risk Analysis

# COMMAND ----------

# Current weather risk by location
print("=== Current Weather Risk by Location ===")
latest_month = weather_risk_final.agg(F.max("month")).collect()[0][0]

display(weather_risk_final.filter(F.col("month") == latest_month) \
    .select("location_name", "state", "location_type", "tavg", "prcp", "snow", 
            "weather_disruption_index", "disruption_risk_level") \
    .orderBy(F.desc("weather_disruption_index")))

# COMMAND ----------

# Weather risk by location type
print("\n=== Average Weather Risk by Location Type ===")
display(weather_risk_final \
    .filter(F.col("month") >= F.add_months(F.current_date(), -12)) \
    .groupBy("location_type") \
    .agg(
        F.avg("weather_disruption_index").alias("avg_disruption_index"),
        F.sum("storm_event_count").alias("total_storm_events"),
        F.avg("extreme_heat_days").alias("avg_heat_days"),
        F.avg("extreme_cold_days").alias("avg_cold_days")
    ) \
    .orderBy(F.desc("avg_disruption_index")))

# COMMAND ----------

# Seasonal weather patterns
print("\n=== Seasonal Weather Patterns (Last 12 Months) ===")
display(weather_risk_final \
    .filter(F.col("month") >= F.add_months(F.current_date(), -12)) \
    .groupBy("month_date") \
    .agg(
        F.avg("weather_disruption_index").alias("avg_disruption_index"),
        F.avg("tavg").alias("avg_temp"),
        F.sum("prcp").alias("total_precip"),
        F.sum("snow").alias("total_snow")
    ) \
    .orderBy("month_date"))

# COMMAND ----------

# High-risk weather events
print("\n=== High Weather Risk Events ===")
display(weather_risk_final \
    .filter(F.col("disruption_risk_level") == "HIGH") \
    .select("month", "location_name", "state", "weather_disruption_index", 
            "extreme_heat_days", "extreme_cold_days", "storm_event_count", "snow") \
    .orderBy(F.desc("month"), F.desc("weather_disruption_index")) \
    .limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Table is in Unity Catalog: `supply_chain.silver.weather_risk_monthly`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Proceed to transformation notebooks to build unified demand signals
# MAGIC 2. Run `01_unified_demand_signals` to combine all data sources
# MAGIC 3. Run `02_dod_metrics_inputs` to calculate DoD metric inputs