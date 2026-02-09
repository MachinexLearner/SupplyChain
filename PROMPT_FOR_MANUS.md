# Prompt for Manus AI: Supply Chain Platform – Databricks Production Deployment

Copy the text below (from "---" to the end) and give it to Manus as the main instruction.

---

You are helping productionize a **Databricks supply chain analytics platform** (defense contracts, subawards, risk signals, demand forecasting, DoD metrics, and AI agent tools). The notebooks are in Databricks .py format and already run in POC mode with synthetic data. Your job is to make them **production-ready for Databricks** and improve them for real data and deployment.

**Context – read first:**
- **`notebooks/README.md`** – Run order (DAG), table lineage (raw → bronze → silver → gold), data mode (POC vs production), production checklist, and “For Manus AI” section.
- Each notebook has an **Executive summary** and, where relevant, **Depends on** and **Production** notes in the first markdown cell.

**Constraints (do not break):**
- **Unity Catalog:** `supply_chain`; **schemas:** `raw`, `bronze`, `silver`, `gold`, `models`. Do **not** rename these or change table names that downstream notebooks or jobs depend on (see README notebook index).
- **Run order:** 00_setup_catalog → ingestion 01–08 → transformation 01–02 → forecasting 01–02 → agents 01. Keep this DAG valid.
- Notebooks are **Databricks notebook source** (.py) with `# COMMAND ----------` cell boundaries and `# MAGIC %md` for markdown. Preserve that format.

**Tasks:**

1. **Production data wiring**
   - **Ingestion:** Where notebooks use synthetic data, implement or complete the **production** path so real data can be used:
     - **01_usaspending_ingestion:** In production mode, load real USAspending bulk award data (from a path or via download from https://www.usaspending.gov/download_center/award_data_archive). Keep output schema aligned with `raw.usa_spending_awards` and `bronze.oshkosh_prime_award_actions`.
     - **02_fpds_ingestion:** Production path already supports a CSV path via widget `fpds_csv_path` or env `FPDS_CSV_PATH`; ensure column mapping matches FPDS export and bronze schema.
     - **03–08:** Add or complete production branches where still synthetic-only (e.g. subawards, SAM, GDELT, tariff/trade, commodity, weather) using the URLs/sources noted in each notebook’s Production section. Prefer widgets (e.g. `data_mode`, path widgets) and env vars (e.g. `DATA_MODE`, `*_PATH`) for job-friendly configuration.
   - Use **idempotent** writes: for incremental loads, replace `mode("overwrite")` with append/merge where appropriate; document which tables are full refresh vs incremental.

2. **Databricks deployment**
   - Add or adjust **Databricks Jobs** (or workflow definitions): at least one job that runs the full pipeline in order (catalog setup once, then ingestion → transformation → forecasting). Optionally separate jobs for ingestion, transformation, and forecasting with correct dependencies.
   - Ensure **cluster/config** assumptions are explicit (e.g. Spark version, Unity Catalog enabled, any init scripts for `%pip install`). Prefer cluster libraries or job-level libraries over notebook `%pip` where possible for production.
   - **Secrets:** Do not hardcode credentials. Use Databricks secrets (e.g. `dbutils.secrets.get`) or environment variables for any API keys or storage paths; document required secret names in README or a CONFIG.md.

3. **Improvements**
   - **Error handling:** Add clear error messages and, where useful, retries or checkpointing for long-running ingestion steps.
   - **Observability:** Ensure key tables are written with useful comments or Delta table properties; add a simple data-quality check (e.g. row counts or null checks) in at least one ingestion notebook and in the unified demand signals notebook.
   - **Documentation:** Update `notebooks/README.md` with any new widgets, env vars, or job parameters; keep the “For Manus AI” and production checklist accurate.

**Deliverables:**
- All ingestion notebooks support production data (real sources or clear stubs with instructions).
- At least one defined Databricks Job (or equivalent workflow) that runs the pipeline end-to-end.
- README (and optional CONFIG.md) updated with deployment steps, required secrets, and any new configuration.
- No breaking changes to catalog name, schema names, or table names used by transformation, forecasting, or agents.

Work in the repo under `supply_chain_platform/notebooks` and any job/workflow definitions you add (e.g. in a `jobs/` or `workflows/` folder or in README). Prefer minimal, clear changes so the platform stays understandable for management and maintainable for future updates.
