# New Ingestion Notebooks (09–13) – Summary Note

**Commodity + IMF:** Combined in **07_commodity_ingestion** (raw/bronze `commodity_prices` with `source` = `yfinance_commodity` or `imf_commodity_prices`; silver `commodity_prices_monthly` from yfinance only). No separate IMF notebook.

## API Endpoints Used

| Notebook | Source | Endpoint / URL |
|----------|--------|----------------|
| **09_worldbank_wdi_ingestion** | World Bank WDI | `https://api.worldbank.org/v2/country/all/indicator/{CODE}?format=json&date=2000:2025&per_page=10000` |
| **10_worldbank_wgi_ingestion** | World Bank WGI | Same base; indicators: CC.EST, GE.EST, PV.EST, RQ.EST, RL.EST, VA.EST |
| **07_commodity_ingestion** | yfinance + IMF PCPS | yfinance (Yahoo Finance); IMF: `https://api.imf.org/data/PCPS?format=jsondata`, `https://data.imf.org/regular.aspx?key=60972224&format=csv` (optional, fails gracefully) |
| **12_nyfed_gscpi_ingestion** | NY Fed GSCPI | Tries: `.../medialibrary/.../gscpi_data.csv`, `.../resources.../gscpi_data.csv`, `.../gscpi/data/gscpi_data.csv` (see notebook `GSCPI_URLS`) |
| **13_wto_trade_barometer_ingestion** | WTO | Tries: `https://stats.wto.org/api/v1/barometer`, `https://api.wto.org/timeseries/v1/data`; optional `WTO_API_KEY` (env or spark.conf `wto.api.key`) |

## Idempotency

- **Merge key:** `(source, indicator_code, country_code, as_of_date)` for every table.
- **Mechanism:** Delta merge (not overwrite). Each notebook:
  1. Creates the raw (and bronze) table with `CREATE TABLE IF NOT EXISTS` and an explicit schema.
  2. Builds a Spark DataFrame from the fetched data with the standard columns.
  3. Uses `DeltaTable.forName(spark, TABLE).alias("t").merge(df.alias("s"), merge_condition).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()`.
- Re-running a notebook updates existing rows for the same key and inserts only new keys; no duplicate rows for the same `(source, indicator_code, country_code, as_of_date)`.

## Indicator Codes Selected

### WDI (09)
Curated set only (not all WDI series):

- **Manufacturing / industrial:** `NV.IND.MANF.CD` (manufacturing value added), `SL.IND.EMPL.ZS` (employment in industry)
- **Trade:** `NE.EXP.GNFS.CD` (exports), `NE.IMP.GNFS.CD` (imports)
- **Energy:** `EG.USE.PCAP.KG.OE` (energy use per capita)
- **Inflation:** `FP.CPI.TOTL` (CPI)
- **Economy:** `NY.GDP.MKTP.CD` (GDP)

### WGI (10)
All six governance dimensions (estimate series):

- `CC.EST` (Control of Corruption), `GE.EST` (Government Effectiveness), `PV.EST` (Political Stability), `RQ.EST` (Regulatory Quality), `RL.EST` (Rule of Law), `VA.EST` (Voice and Accountability)

### Commodity + IMF (07)
**07_commodity_ingestion** ingests both yfinance (defense tickers: CL=F, GC=F, etc.) and IMF PCPS (monthly indices when available). Raw/bronze: `supply_chain.raw.commodity_prices`, `supply_chain.bronze.commodity_prices` with `source` = `yfinance_commodity` or `imf_commodity_prices`. Silver `commodity_prices_monthly` is built from yfinance only (cost-pressure metrics). IMF codes depend on API/CSV response; no separate IMF table.

### NY Fed (12)
Single indicator: **GSCPI** (Global Supply Chain Pressure Index), monthly.

### WTO (13)
Single indicator: **WTO_GOODS_BAROMETER** (quarterly), when available via API; otherwise the notebook fails with a clear message and no scraping.

## Shared Helper

- **ingestion_utils.py** provides: `safe_get(url, timeout=..., retries=3, backoff=2.0)`, `parse_json(text)`, `normalize_indicator_row(...)` for the standard schema. Each notebook tries to import it and falls back to inline definitions if the module is not on the path.

## Output Tables (Stable Names)

- Raw: `supply_chain.raw.worldbank_wdi`, `supply_chain.raw.worldbank_wgi`, `supply_chain.raw.commodity_prices` (yfinance + IMF by `source`), `supply_chain.raw.nyfed_gscpi`, `supply_chain.raw.wto_trade_barometer`
- Bronze: `supply_chain.bronze.worldbank_wdi`, `supply_chain.bronze.worldbank_wgi`, `supply_chain.bronze.commodity_prices`, `supply_chain.bronze.nyfed_gscpi`, `supply_chain.bronze.wto_trade_barometer`
- Silver: `supply_chain.silver.commodity_prices_monthly` (from 07, yfinance-only cost-pressure view)

Standard indicator tables use: `source`, `ingested_at`, `as_of_date`, `country_code`, `indicator_code`, `indicator_name`, `value`, `unit`, `frequency`, `raw_payload`.
