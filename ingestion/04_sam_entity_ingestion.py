# Databricks notebook source
# MAGIC %md
# MAGIC # SAM.gov Entity Data Ingestion (Live API)
# MAGIC
# MAGIC **Executive summary:** Fetches supplier entity and location data from the **SAM.gov Entity Management API v3** into raw and silver (supplier geolocations, distance to facilities). Searches by defense-relevant NAICS codes to find active registrants.
# MAGIC
# MAGIC **Data Source**: SAM.gov Entity Management API v3 — https://open.gsa.gov/api/entity-api/
# MAGIC
# MAGIC **Target Tables** (Unity Catalog):
# MAGIC - `supply_chain.raw.sam_entity_export` — Raw SAM entity data
# MAGIC - `supply_chain.silver.supplier_geolocations` — Geocoded supplier locations
# MAGIC
# MAGIC **API Key:** Obtain a free API key at https://sam.gov/content/entity-information — set via widget, env var `SAM_API_KEY`, or Databricks secret scope `supply_chain/sam_api_key`.

# COMMAND ----------

# MAGIC %pip install requests pandas pyarrow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Widget for API key (can also be set via env var or Databricks secrets)
dbutils.widgets.text("sam_api_key", "SAM-45654ff9-9605-4e0a-bfe8-fb30e0d939e3", "SAM.gov API key (or set SAM_API_KEY env var)")

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
RAW_TABLE = f"{CATALOG}.raw.sam_entity_export"
SILVER_TABLE = f"{CATALOG}.silver.supplier_geolocations"

# Oshkosh Defense facility locations for distance calculations
OSHKOSH_FACILITIES = [
    {'name': 'Oshkosh HQ', 'city': 'Oshkosh', 'state': 'WI', 'lat': 44.0247, 'lon': -88.5426},
    {'name': 'Oshkosh Defense', 'city': 'Oshkosh', 'state': 'WI', 'lat': 44.0247, 'lon': -88.5426},
    {'name': 'JLG Industries', 'city': 'McConnellsburg', 'state': 'PA', 'lat': 39.9326, 'lon': -77.9967},
    {'name': 'Pierce Manufacturing', 'city': 'Appleton', 'state': 'WI', 'lat': 44.2619, 'lon': -88.4154},
]

# Region mapping for geopolitical analysis
REGION_MAPPING = {
    # Americas
    'USA': 'AMERICAS', 'CAN': 'AMERICAS', 'MEX': 'AMERICAS', 'BRA': 'AMERICAS',
    # Europe
    'GBR': 'EUROPE', 'DEU': 'EUROPE', 'FRA': 'EUROPE', 'ITA': 'EUROPE', 'POL': 'EUROPE',
    'ESP': 'EUROPE', 'NLD': 'EUROPE', 'BEL': 'EUROPE', 'AUT': 'EUROPE', 'CHE': 'EUROPE',
    'SWE': 'EUROPE', 'NOR': 'EUROPE', 'DNK': 'EUROPE', 'FIN': 'EUROPE', 'CZE': 'EUROPE',
    # Middle East
    'ISR': 'MIDEAST', 'SAU': 'MIDEAST', 'ARE': 'MIDEAST', 'KWT': 'MIDEAST', 'QAT': 'MIDEAST',
    'JOR': 'MIDEAST', 'IRQ': 'MIDEAST', 'TUR': 'MIDEAST',
    # Indo-Pacific
    'JPN': 'INDO_PACIFIC', 'KOR': 'INDO_PACIFIC', 'AUS': 'INDO_PACIFIC', 'TWN': 'INDO_PACIFIC',
    'SGP': 'INDO_PACIFIC', 'THA': 'INDO_PACIFIC', 'PHL': 'INDO_PACIFIC', 'IND': 'INDO_PACIFIC',
    'NZL': 'INDO_PACIFIC', 'MYS': 'INDO_PACIFIC', 'IDN': 'INDO_PACIFIC',
    # Africa
    'ZAF': 'AFRICA', 'EGY': 'AFRICA', 'MAR': 'AFRICA', 'NGA': 'AFRICA', 'KEN': 'AFRICA',
    # China (separate for risk analysis)
    'CHN': 'CHINA',
}

# Defense-relevant NAICS codes
DEFENSE_NAICS_CODES = {
    '336120': 'Heavy Duty Truck Manufacturing',
    '336211': 'Motor Vehicle Body Manufacturing',
    '336992': 'Military Armored Vehicle Manufacturing',
    '336999': 'Other Transportation Equipment Manufacturing',
    '332994': 'Small Arms, Ordnance, and Ordnance Accessories Manufacturing',
}

# Subsystem category classification based on NAICS/PSC
NAICS_TO_SUBSYSTEM = {
    '336120': 'POWERTRAIN',
    '336211': 'ARMOR',
    '336992': 'ARMOR',
    '336999': 'ELECTRONICS',
    '332994': 'ARMOR',
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAM.gov Entity API Fetch
# MAGIC
# MAGIC Queries the SAM.gov Entity Management API v3 for active defense-related registrants
# MAGIC by NAICS code with `purposeOfRegistrationCode=Z2` (Federal Assistance awards + contracts).

# COMMAND ----------

def _get_sam_api_key() -> str:
    """
    Resolve SAM.gov API key from (in priority order):
      1. Databricks widget
      2. Environment variable SAM_API_KEY
      3. Databricks secret scope supply_chain/sam_api_key
    Raises ValueError with instructions if none found.
    """
    # 1. Widget value
    try:
        key = dbutils.widgets.get("sam_api_key").strip()
        if key:
            return key
    except Exception:
        pass

    # 2. Environment variable
    key = os.environ.get("SAM_API_KEY", "").strip()
    if key:
        return key

    # 3. Databricks secrets
    try:
        key = dbutils.secrets.get(scope="supply_chain", key="sam_api_key").strip()
        if key:
            return key
    except Exception:
        pass

    raise ValueError(
        "No SAM.gov API key found. Provide one via:\n"
        "  1. The 'sam_api_key' widget at the top of this notebook\n"
        "  2. Environment variable SAM_API_KEY\n"
        "  3. Databricks secret scope 'supply_chain' key 'sam_api_key'\n\n"
        "Get a free API key at: https://sam.gov/content/entity-information"
    )

# COMMAND ----------

def fetch_sam_entities() -> pd.DataFrame:
    """
    Fetch active entity registrations from SAM.gov Entity Management API v3
    for defense-relevant NAICS codes.

    Returns a pandas DataFrame with one row per entity, mapped to the
    sam_entity_export schema.
    """
    api_key = _get_sam_api_key()
    base_url = "https://api.sam.gov/entity-information/v3/entities"

    all_entities = []
    seen_ueis = set()  # deduplicate across NAICS queries

    for naics_code, naics_desc in DEFENSE_NAICS_CODES.items():
        print(f"Fetching entities for NAICS {naics_code} ({naics_desc}) ...")
        page = 0
        total_fetched_for_naics = 0

        while True:
            params = {
                "api_key": api_key,
                "registrationStatus": "A",
                "naicsCode": naics_code,
                "purposeOfRegistrationCode": "Z2",
                "includeSections": "entityRegistration,coreData",
                "page": page,
                "size": 100,
            }

            # Retry logic — 3 attempts with exponential backoff
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(base_url, params=params, timeout=30)
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
                print(f"  Skipping NAICS {naics_code} after retries (last status: {getattr(response, 'status_code', 'N/A')})")
                break

            data = response.json()
            entities = data.get("entityData", [])
            total_records = data.get("totalRecords", 0)

            if not entities:
                break

            for entity in entities:
                reg = entity.get("entityRegistration", {})
                core = entity.get("coreData", {})
                phys_addr = core.get("physicalAddress", {})
                entity_info = core.get("entityInformation", {})

                uei = reg.get("ueiSAM")
                if not uei or uei in seen_ueis:
                    continue
                seen_ueis.add(uei)

                # Map business types list to string
                biz_types = reg.get("businessTypes", [])
                biz_type_str = ", ".join(biz_types) if isinstance(biz_types, list) else str(biz_types) if biz_types else None

                # Classify subsystem from NAICS
                primary_naics = entity_info.get("primaryNaics", naics_code)
                subsystem = NAICS_TO_SUBSYSTEM.get(str(primary_naics), "OTHER")

                # Classify company size from SAM business type flags
                company_size = None
                if biz_type_str:
                    bt_lower = biz_type_str.lower()
                    if "small" in bt_lower:
                        company_size = "SMALL"
                    elif "large" in bt_lower:
                        company_size = "LARGE"
                    else:
                        company_size = "MEDIUM"

                all_entities.append({
                    "uei": uei,
                    "cage_code": reg.get("cageCode"),
                    "legal_business_name": reg.get("legalBusinessName"),
                    "dba_name": reg.get("dbaName"),
                    "physical_address_line_1": phys_addr.get("addressLine1"),
                    "physical_address_city": phys_addr.get("city"),
                    "physical_address_state_or_province": phys_addr.get("stateOrProvinceCode"),
                    "physical_address_zip_postal_code": phys_addr.get("zipCode"),
                    "physical_address_country_code": phys_addr.get("countryCode"),
                    "entity_start_date": reg.get("registrationDate"),
                    "entity_expiration_date": reg.get("registrationExpirationDate"),
                    "activation_date": reg.get("activationDate"),
                    "business_type": biz_type_str,
                    "entity_structure": entity_info.get("entityStructureDesc"),
                    "organization_type": entity_info.get("organizationStructureDesc"),
                    "naics_code_primary": str(primary_naics) if primary_naics else naics_code,
                    "subsystem_category": subsystem,
                    "company_size": company_size,
                    "sam_extract_code": "A",
                })
                total_fetched_for_naics += 1

            print(f"  Page {page}: got {len(entities)} entities (total so far: {total_fetched_for_naics}, API reports {total_records} total)")

            # Next page or done
            if total_fetched_for_naics >= total_records or len(entities) < 100:
                break
            page += 1
            time.sleep(0.5)  # polite pacing between pages

    print(f"\nTotal unique entities fetched: {len(all_entities)}")
    return pd.DataFrame(all_entities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Geocoding
# MAGIC
# MAGIC Use OpenStreetMap Nominatim to geocode entity addresses.
# MAGIC Rate-limited to 1 request/second per Nominatim usage policy.

# COMMAND ----------

def geocode_with_nominatim(address: str, city: str, state: str, country: str) -> tuple:
    """
    Geocode an address using OpenStreetMap Nominatim (free, no API key).
    
    Note: In production, respect Nominatim usage policy (1 request/second).
    
    Args:
        address: Street address
        city: City name
        state: State/province
        country: Country code
    
    Returns:
        Tuple of (latitude, longitude) or (None, None) if not found
    """
    try:
        query = f"{city}, {state}, {country}"
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': query,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'DatabricksSupplyChainPlatform/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        
        return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None

# COMMAND ----------

def geocode_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add latitude/longitude columns to entity DataFrame using Nominatim.
    Respects 1 request/second rate limit.
    """
    latitudes = []
    longitudes = []
    total = len(df)

    for idx, row in df.iterrows():
        city = row.get("physical_address_city", "")
        state = row.get("physical_address_state_or_province", "")
        country = row.get("physical_address_country_code", "")

        if city and state:
            lat, lon = geocode_with_nominatim(
                row.get("physical_address_line_1", ""),
                city, state, country
            )
            latitudes.append(lat)
            longitudes.append(lon)
            if (idx + 1) % 25 == 0:
                print(f"  Geocoded {idx + 1}/{total} ...")
            # Nominatim policy: max 1 request per second
            time.sleep(1.0)
        else:
            latitudes.append(None)
            longitudes.append(None)

    df["latitude"] = latitudes
    df["longitude"] = longitudes
    geocoded = sum(1 for lat in latitudes if lat is not None)
    print(f"Geocoding complete: {geocoded}/{total} entities resolved")
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

# Fetch entities from SAM.gov API
print("Fetching entity data from SAM.gov Entity Management API v3 ...")
sam_df = fetch_sam_entities()
print(f"Fetched {len(sam_df)} entity records")

# COMMAND ----------

# Geocode entity addresses
print("Geocoding entity addresses via Nominatim (1 req/sec) ...")
sam_df = geocode_entities(sam_df)

# COMMAND ----------

# Convert to Spark DataFrame
spark_sam = spark.createDataFrame(sam_df)

# Display schema
print("SAM Entity Schema:")
spark_sam.printSchema()

# COMMAND ----------

# Display sample
display(spark_sam.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Raw Layer

# COMMAND ----------

# Unity Catalog setup
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")

# Save to raw layer (Unity Catalog)
spark_sam.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(RAW_TABLE)

print(f"Saved {spark_sam.count()} records to {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Geocoding and Distance Calculations

# COMMAND ----------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km).
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
    
    Returns:
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in km
    r = 6371
    
    return c * r

# Register as UDF
haversine_udf = F.udf(haversine_distance, DoubleType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Silver Layer - Supplier Geolocations

# COMMAND ----------

# Read raw SAM data (Unity Catalog)
sam_raw = spark.table(RAW_TABLE)

# Add region group based on country
def get_region_group(country_code):
    return REGION_MAPPING.get(country_code, 'OTHER')

region_udf = F.udf(get_region_group, StringType())

# Calculate distance to nearest Oshkosh facility
# Using Oshkosh HQ as primary reference point
oshkosh_hq_lat = 44.0247
oshkosh_hq_lon = -88.5426

# Build silver layer
supplier_geolocations = sam_raw \
    .withColumnRenamed("legal_business_name", "supplier_name") \
    .withColumnRenamed("physical_address_city", "city") \
    .withColumnRenamed("physical_address_state_or_province", "state") \
    .withColumnRenamed("physical_address_country_code", "country") \
    .withColumnRenamed("latitude", "lat") \
    .withColumnRenamed("longitude", "lon") \
    .withColumn("region_group", region_udf(F.col("country"))) \
    .withColumn(
        "distance_to_nearest_oshkosh_facility_km",
        haversine_udf(
            F.col("lat"),
            F.col("lon"),
            F.lit(oshkosh_hq_lat),
            F.lit(oshkosh_hq_lon)
        )
    ) \
    .select(
        "supplier_name",
        "uei",
        "cage_code",
        "city",
        "state",
        "country",
        "lat",
        "lon",
        "region_group",
        "distance_to_nearest_oshkosh_facility_km",
        "subsystem_category",
        "company_size",
        "naics_code_primary",
    ) \
    .withColumn("ingestion_timestamp", F.current_timestamp()) \
    .withColumn("source_system", F.lit("sam_gov_api_v3"))

# COMMAND ----------

# Display sample
display(supplier_geolocations.limit(15))

# COMMAND ----------

# Save to silver layer (Unity Catalog)
supplier_geolocations.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE)

print(f"Saved {supplier_geolocations.count()} records to {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supplier Geography Analysis

# COMMAND ----------

# Suppliers by region
print("=== Suppliers by Region ===")
display(supplier_geolocations.groupBy("region_group").agg(
    F.count("*").alias("supplier_count"),
    F.avg("distance_to_nearest_oshkosh_facility_km").alias("avg_distance_km")
).orderBy(F.desc("supplier_count")))

# COMMAND ----------

# Suppliers by subsystem and region
print("\n=== Suppliers by Subsystem and Region ===")
display(supplier_geolocations.groupBy("subsystem_category", "region_group").agg(
    F.count("*").alias("supplier_count")
).orderBy("subsystem_category", "region_group"))

# COMMAND ----------

# Distance distribution
print("\n=== Supplier Distance Distribution ===")
display(supplier_geolocations.select(
    "supplier_name",
    "city",
    "state",
    "country",
    "region_group",
    "distance_to_nearest_oshkosh_facility_km",
    "subsystem_category"
).orderBy(F.desc("distance_to_nearest_oshkosh_facility_km")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supply Chain Risk by Geography

# COMMAND ----------

# Calculate geographic risk scores
geo_risk = supplier_geolocations.groupBy("region_group").agg(
    F.count("*").alias("supplier_count"),
    F.avg("distance_to_nearest_oshkosh_facility_km").alias("avg_distance_km"),
    F.max("distance_to_nearest_oshkosh_facility_km").alias("max_distance_km")
).withColumn(
    "geographic_risk_level",
    F.when(F.col("region_group") == "CHINA", "CRITICAL")
     .when(F.col("region_group").isin(["MIDEAST", "AFRICA"]), "HIGH")
     .when(F.col("region_group").isin(["INDO_PACIFIC", "EUROPE"]), "MODERATE")
     .when(F.col("region_group") == "AMERICAS", "LOW")
     .otherwise("UNKNOWN")
)

print("=== Geographic Supply Chain Risk ===")
display(geo_risk.orderBy("supplier_count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC Tables written to Unity Catalog: `supply_chain.raw.sam_entity_export`, `supply_chain.silver.supplier_geolocations`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `06_tariff_trade_ingestion` for trade risk data
# MAGIC 2. Run `07_commodity_ingestion` for commodity price data
# MAGIC 3. Run `08_weather_ingestion` for weather risk data
