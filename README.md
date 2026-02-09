# Supply Chain Platform Notebooks

**Purpose:** Databricks notebooks for defense supply chain analytics—contracts, subawards, risk signals, demand signals, DoD metrics, and forecasting. Ready for **Manus AI** to modify and improve for Databricks deployment and for **management** to understand data flow and business value.

---

## Executive Summary (Management)

| Area | What It Does | Business Value |
|------|--------------|----------------|
| **Ingestion** | Loads federal contract data (USAspending, FPDS), subawards, supplier locations (SAM.gov), tariff/trade events, commodity prices, and weather risk. | Single source of truth for contracts, suppliers, and risk; supports planning and reporting. |
| **Transformation** | Joins all signals into monthly demand and risk indices; computes DoD metrics (RO, AAO, safety stock, days of supply). | Unified view for forecasting and DoD compliance. |
| **Forecasting** | Prophet and ARIMA models for monthly demand; MLflow tracking and Unity Catalog model registry. | Demand planning and capacity decisions. |
| **Agents** | AI tools for forecasts, anomaly detection, scenario analysis (geo, tariff, weather), DoD metrics, commodity prices. | Self-serve analytics and what-if analysis. |

**Data lineage:** Raw (source files) → Bronze (cleaned/validated) → Silver (enriched/joined) → Gold (metrics, demand signals, forecasts).

---

## Upload to Databricks

Notebooks are provided in two formats:

- **`.py` (Databricks source format)** — Use with **Repos** (Git sync) or **Workspace → Import → Python**. Databricks treats `# COMMAND ----------` as cell boundaries and `# MAGIC %md` as markdown.
- **`.ipynb` (Jupyter)** — For **direct upload**: **Workspace → Import → Jupyter Notebook**, then select the `.ipynb` files. Re-generate `.ipynb` from `.py` anytime with:
  ```bash
  python scripts/convert_py_to_ipynb.py
  ```
  Or convert a single file: `python scripts/convert_py_to_ipynb.py ingestion/09_worldbank_wdi_ingestion.py`

---

## Run Order (DAG)

Run in this order for a full pipeline:

1. **`ingestion/00_setup_catalog.py`** — Create Unity Catalog `supply_chain` and schemas (`raw`, `bronze`, `silver`, `gold`, `models`). **Run once first.**
2. **Ingestion** (any order after 00):  
   `01_usaspending_ingestion` → `02_fpds_ingestion` → `03_subaward_ingestion` → `04_sam_entity_ingestion` → `06_tariff_trade_ingestion` → `07_commodity_ingestion` → `08_weather_ingestion` → `09_worldbank_wdi_ingestion` → `10_worldbank_wgi_ingestion` → `12_nyfed_gscpi_ingestion` → `13_wto_trade_barometer_ingestion`
3. **Transformation:**  
   `transformation/01_unified_demand_signals.py` → `transformation/02_dod_metrics_inputs.py`
4. **Forecasting:**  
   `forecasting/01_prophet_forecasting.py`, `forecasting/02_arima_forecasting.py`
5. **Agents:**  
   `agents/01_agent_tools.py` (uses gold tables and forecasts)

---

## Run the full pipeline end-to-end in Databricks

### Prerequisites

- **Cluster:** A cluster (or job cluster) with Unity Catalog enabled and access to the `supply_chain` catalog. For ingestion notebooks that call external APIs (World Bank, NY Fed, etc.), ensure the cluster has outbound network access.
- **Notebooks in workspace:** Upload or sync the notebooks (e.g. via **Workspace → Import** or **Repos**). Note the workspace path to the folder (e.g. `/Workspace/Users/you@company.com/supply_chain_platform/notebooks` or `/Repos/your-org/supply_chain_platform/notebooks`).
- **One-time setup:** Run `ingestion/00_setup_catalog` once (manually or as the first job task). Set the `catalog_location` widget or env to your S3/ADLS path if required.

### Option 1: Databricks Workflows job (recommended)

1. Go to **Workflows → Jobs → Create job**.
2. Name the job (e.g. `supply_chain_full_pipeline`).
3. Add tasks in this order, setting **Dependencies** so each task runs after the previous one:
   - **Task 1:** `setup` — Notebook: `ingestion/00_setup_catalog` (or full path under your workspace).
   - **Task 2:** `usaspending` — Notebook: `ingestion/01_usaspending_ingestion`, depends on `setup`.
   - **Task 3:** `fpds` — Notebook: `ingestion/02_fpds_ingestion`, depends on `usaspending`.
   - **Task 4:** `subaward` — Notebook: `ingestion/03_subaward_ingestion`, depends on `fpds`.
   - **Task 5:** `sam` — Notebook: `ingestion/04_sam_entity_ingestion`, depends on `subaward`.
   - **Task 6:** `tariff` — Notebook: `ingestion/06_tariff_trade_ingestion`, depends on `sam`.
   - **Task 7:** `commodity` — Notebook: `ingestion/07_commodity_ingestion`, depends on `tariff`.
   - **Task 8:** `weather` — Notebook: `ingestion/08_weather_ingestion`, depends on `commodity`.
   - **Task 9:** `worldbank_wdi` — Notebook: `ingestion/09_worldbank_wdi_ingestion`, depends on `weather`.
   - **Task 10:** `worldbank_wgi` — Notebook: `ingestion/10_worldbank_wgi_ingestion`, depends on `worldbank_wdi`.
   - **Task 11:** `nyfed_gscpi` — Notebook: `ingestion/12_nyfed_gscpi_ingestion`, depends on `worldbank_wgi`.
   - **Task 12:** `wto_barometer` — Notebook: `ingestion/13_wto_trade_barometer_ingestion`, depends on `nyfed_gscpi`.
   - **Task 13:** `unified_signals` — Notebook: `transformation/01_unified_demand_signals`, depends on `wto_barometer`.
   - **Task 14:** `dod_metrics` — Notebook: `transformation/02_dod_metrics_inputs`, depends on `unified_signals`.
   - **Task 15:** `prophet` — Notebook: `forecasting/01_prophet_forecasting`, depends on `dod_metrics`.
   - **Task 16:** `arima` — Notebook: `forecasting/02_arima_forecasting`, depends on `dod_metrics`.
4. Set the **Cluster** for the job (existing cluster or job cluster). Use the same cluster for all tasks, or separate clusters per task if you prefer.
5. **Run now** or **Schedule** (e.g. daily/weekly for ingestion, then transformation and forecasting).

**Using the job definition file (CLI):** Create the job from the JSON definition, then run it:

1. Edit `notebooks/jobs/supply_chain_full_pipeline.json` and replace `/Workspace/SupplyChain/notebooks` with your actual notebook path (e.g. `/Workspace/Users/you@company.com/supply_chain_platform/notebooks` or `/Repos/your-org/supply_chain_platform/notebooks`).
2. With the [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html) configured (`databricks configure`), run:
   ```bash
   databricks jobs create --json-file notebooks/jobs/supply_chain_full_pipeline.json
   ```
3. In the UI, open the job and click **Run now**, or attach a schedule.

### Option 2: Run notebooks manually in order

1. Run **`ingestion/00_setup_catalog`** once.
2. Run the ingestion notebooks **01 → 02 → 03 → 04 → 06 → 07 → 08 → 09 → 10 → 12 → 13** (you can run 01–13 in any order after 00 if you prefer).
3. Run **`transformation/01_unified_demand_signals`**, then **`transformation/02_dod_metrics_inputs`**.
4. Run **`forecasting/01_prophet_forecasting`** and **`forecasting/02_arima_forecasting`** (order between the two doesn’t matter).
5. Optionally run **`agents/01_agent_tools`** for interactive or on-demand use.

---

## Table Lineage (Unity Catalog)

| Layer | Example Tables | Source |
|-------|----------------|--------|
| **raw** | `usa_spending_awards`, `fpds_contracts`, … | Bulk downloads / APIs |
| **bronze** | `oshkosh_prime_award_actions`, `oshkosh_subawards`, `fpds_contracts` | Raw + filters + metadata |
| **silver** | `supplier_geolocations`, `commodity_prices_monthly`, `weather_risk_monthly`, `trade_tariff_risk_events` | Bronze + enrichment |
| **gold** | `oshkosh_monthly_demand_signals`, `dod_metrics_inputs_monthly`, `trade_tariff_risk_monthly`, `prophet_forecasts`, `arima_forecasts` | Silver + aggregations / models |

---

## Data Mode: POC vs Production

- **POC (default):** Notebooks use **synthetic data** or fallbacks (e.g., commodity/weather try real APIs then synthetic) so the pipeline runs without external credentials or file uploads.
- **Production:** Set widget `data_mode` = `production` (where implemented) and/or wire **real data**:
  - USAspending: bulk download from https://www.usaspending.gov/download_center/award_data_archive → upload to cloud storage or mount; point notebook at path.
  - FPDS: export from https://www.fpds.gov (Advanced Search) → CSV to cloud storage.
  - Tariff/trade: synthetic or USITC/WTO exports.
  - SAM.gov: Entity Management Public V2 export.
  - Commodity: yfinance (real by default when available).
  - Weather: Meteostat (real by default when available).

**For Manus AI:** Replace synthetic generators with real API/ETL logic when `data_mode == "production"`; keep catalog and table names unchanged for downstream notebooks.

---

## Production Deployment Checklist

- [ ] Run `00_setup_catalog` with correct `catalog_location` (e.g. S3/ADLS path covered by external location).
- [ ] Create external location and grant `supply_chain` catalog access.
- [ ] For production: switch ingestion notebooks to real sources (widget `data_mode` or env `DATA_MODE=production`).
- [ ] Schedule jobs: ingestion (e.g. daily/weekly) → transformation → forecasting; optionally run agents on-demand or on schedule.
- [ ] Use cluster policies and Unity Catalog permissions for production workloads.
- [ ] Register models (Prophet/ARIMA) in Unity Catalog; use alias `champion` for serving.

---

## Notebook Index

| Notebook | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `ingestion/00_setup_catalog` | Create catalog & schemas | Widget: catalog_location | `supply_chain` catalog, schemas |
| `ingestion/01_usaspending_ingestion` | USAspending awards | Bulk download or synthetic | `raw.usa_spending_awards`, `bronze.oshkosh_prime_award_actions` |
| `ingestion/02_fpds_ingestion` | FPDS contracts | Export CSV or synthetic | `raw.fpds_contracts`, `bronze.fpds_contracts` |
| `ingestion/03_subaward_ingestion` | Subawards | USAspending FSRS or synthetic | `raw.usa_spending_subawards`, `bronze.oshkosh_subawards` |
| `ingestion/04_sam_entity_ingestion` | SAM.gov entities | SAM export or synthetic | `raw.sam_entity_export`, `silver.supplier_geolocations` |
| `ingestion/06_tariff_trade_ingestion` | Tariff/trade events | USITC/WTO or synthetic | `raw.trade_tariff_events`, `silver.trade_tariff_risk_events`, `gold.trade_tariff_risk_monthly` |
| `ingestion/07_commodity_ingestion` | Commodity prices | yfinance or synthetic | `silver.commodity_prices_monthly` |
| `ingestion/08_weather_ingestion` | Weather risk | Meteostat or synthetic | `silver.weather_risk_monthly` |
| `transformation/01_unified_demand_signals` | Unified monthly signals | Bronze/silver/gold tables above | `gold.oshkosh_monthly_demand_signals` |
| `transformation/02_dod_metrics_inputs` | DoD metrics | `oshkosh_monthly_demand_signals` | `gold.dod_metrics_inputs_monthly` |
| `forecasting/01_prophet_forecasting` | Prophet demand forecast | Demand signals | `gold.prophet_forecasts`, MLflow + UC model |
| `forecasting/02_arima_forecasting` | ARIMA demand forecast | Demand signals | `gold.arima_forecasts` |
| `agents/01_agent_tools` | AI agent tools | Gold tables, forecasts | Queries and scenario outputs |

---

## For Manus AI

- Notebooks are **Databricks .py format** (cell boundaries: `# COMMAND ----------`).
- **Catalog:** `supply_chain`; **schemas:** `raw`, `bronze`, `silver`, `gold`, `models`. Do not change table names if downstream notebooks or jobs depend on them.
- **Production:** Add or use `data_mode` (dropdown: `poc` | `production`) and branch logic: POC = synthetic; production = real API/ETL. Prefer widgets for interactive runs and environment variables for jobs.
- **Dependencies:** `%pip install` cells at top where needed; avoid DBFS root when disabled—use Unity Catalog and workspace paths.
- **Idempotency:** Writes use `mode("overwrite")`; for incremental loads, replace with append/merge per table.
