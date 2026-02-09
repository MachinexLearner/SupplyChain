# Databricks notebook source
# MAGIC %md
# MAGIC # Prophet Demand Forecasting with MLflow
# MAGIC
# MAGIC **Executive summary:** Trains Prophet demand forecast with fiscal seasonality and risk regressors; logs to MLflow and writes gold forecasts. Management: use for demand planning and capacity; register model in Unity Catalog for production serving.
# MAGIC
# MAGIC **Depends on:** `supply_chain.gold.oshkosh_monthly_demand_signals` (run transformation notebooks first).
# MAGIC
# MAGIC This notebook implements demand forecasting using Facebook Prophet with:
# MAGIC - Defense-specific seasonality (fiscal year, military exercises)
# MAGIC - Exogenous regressors (risk indices, commodity prices)
# MAGIC - MLflow experiment tracking and model registry
# MAGIC
# MAGIC **Model**: Prophet with custom seasonality
# MAGIC **Target**: Monthly demand obligations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Install required packages
%pip install prophet mlflow pandas numpy scikit-learn matplotlib

# COMMAND ----------

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import mlflow.prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
DEMAND_SIGNALS_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"
DOD_METRICS_TABLE = f"{CATALOG}.gold.dod_metrics_inputs_monthly"
FORECAST_OUTPUT_TABLE = f"{CATALOG}.gold.prophet_forecasts"
# When DBFS root is disabled, use a workspace path (e.g. /Workspace/Users/<you>/models/prophet_demand_forecast)
MODEL_PATH = "/models/prophet_demand_forecast"

# MLflow configuration
EXPERIMENT_NAME = "/Shared/supply_chain_platform/experiments/demand_forecasting"

# Forecast parameters
FORECAST_HORIZON_MONTHS = 12
CROSS_VALIDATION_INITIAL = "730 days"  # 2 years initial training
CROSS_VALIDATION_PERIOD = "90 days"    # Retrain every 90 days
CROSS_VALIDATION_HORIZON = "90 days"   # Forecast 90 days ahead

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow Experiment

# COMMAND ----------

# Set up MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Prophet does not support mlflow.prophet.autolog(); parameters/metrics are logged manually below.

print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load demand signals (Unity Catalog)
try:
    demand_spark = spark.table(DEMAND_SIGNALS_TABLE)
    demand_df = demand_spark.toPandas()
    print(f"Loaded {len(demand_df)} demand signal records")
except Exception as e:
    print(f"Error loading demand signals: {e}")
    demand_df = None

# COMMAND ----------

if demand_df is not None:
    # Prepare data for Prophet
    # Prophet requires columns named 'ds' (date) and 'y' (target)
    prophet_df = demand_df[['month', 'total_obligations_usd', 
                            'geo_risk_index', 'tariff_risk_index',
                            'commodity_cost_pressure', 'weather_disruption_index',
                            'combined_risk_index']].copy()
    
    prophet_df.columns = ['ds', 'y', 'geo_risk', 'tariff_risk', 
                          'commodity_pressure', 'weather_risk', 'combined_risk']
    
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df = prophet_df.sort_values('ds')
    
    # Remove rows with zero demand (if any)
    prophet_df = prophet_df[prophet_df['y'] > 0]
    
    print(f"Prepared {len(prophet_df)} records for Prophet")
    print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    print(prophet_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Defense-Specific Events

# COMMAND ----------

# Define military-relevant events and holidays
def create_defense_events(years: list) -> pd.DataFrame:
    """
    Create DataFrame of defense-relevant events that affect demand.
    """
    events = []
    
    for year in years:
        # Federal fiscal year events
        events.append({
            'holiday': 'fiscal_year_start',
            'ds': pd.Timestamp(f'{year}-10-01'),
            'lower_window': 0,
            'upper_window': 14
        })
        events.append({
            'holiday': 'fiscal_year_end',
            'ds': pd.Timestamp(f'{year}-09-30'),
            'lower_window': -14,
            'upper_window': 0
        })
        
        # Major military exercises (approximate dates)
        events.append({
            'holiday': 'spring_exercise',
            'ds': pd.Timestamp(f'{year}-04-15'),
            'lower_window': -7,
            'upper_window': 14
        })
        events.append({
            'holiday': 'fall_exercise',
            'ds': pd.Timestamp(f'{year}-11-15'),
            'lower_window': -7,
            'upper_window': 14
        })
        
        # Budget submission periods
        events.append({
            'holiday': 'budget_submission',
            'ds': pd.Timestamp(f'{year}-02-01'),
            'lower_window': -7,
            'upper_window': 30
        })
    
    return pd.DataFrame(events)

# Create events for relevant years
if demand_df is not None:
    years = list(range(2010, 2028))
    defense_events = create_defense_events(years)
    print(f"Created {len(defense_events)} defense events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Prophet Model

# COMMAND ----------

def train_prophet_model(df: pd.DataFrame, 
                        events: pd.DataFrame = None,
                        include_regressors: bool = True) -> Prophet:
    """
    Train Prophet model with defense-specific configuration.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns
        events: DataFrame with holiday/event definitions
        include_regressors: Whether to include exogenous regressors
    
    Returns:
        Trained Prophet model
    """
    # Initialize Prophet with defense-relevant settings
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # Monthly data, no weekly pattern
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for varying demand
        changepoint_prior_scale=0.05,  # Conservative trend changes
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        interval_width=0.95,  # 95% prediction intervals
        holidays=events
    )
    
    # Add federal fiscal year seasonality (quarterly pattern)
    model.add_seasonality(
        name='fiscal_quarterly',
        period=91.25,  # ~3 months
        fourier_order=5
    )
    
    # Add exogenous regressors if available
    if include_regressors:
        if 'geo_risk' in df.columns:
            model.add_regressor('geo_risk', mode='multiplicative')
        if 'tariff_risk' in df.columns:
            model.add_regressor('tariff_risk', mode='multiplicative')
        if 'commodity_pressure' in df.columns:
            model.add_regressor('commodity_pressure', mode='additive')
        if 'weather_risk' in df.columns:
            model.add_regressor('weather_risk', mode='multiplicative')
    
    # Fit model
    model.fit(df)
    
    return model

# COMMAND ----------

# Train model with MLflow tracking
if demand_df is not None and len(prophet_df) > 24:  # Need at least 2 years of data
    with mlflow.start_run(run_name="prophet_demand_forecast"):
        # Log parameters
        mlflow.log_param("forecast_horizon_months", FORECAST_HORIZON_MONTHS)
        mlflow.log_param("seasonality_mode", "multiplicative")
        mlflow.log_param("include_regressors", True)
        mlflow.log_param("training_samples", len(prophet_df))
        
        # Train model
        print("Training Prophet model...")
        model = train_prophet_model(
            prophet_df,
            events=defense_events,
            include_regressors=True
        )
        
        print("Model trained successfully")
        
        # Log model
        mlflow.prophet.log_model(model, "prophet_model")
        
        # Get run ID for later reference
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Forecasts

# COMMAND ----------

if demand_df is not None:
    # Create future dataframe
    future = model.make_future_dataframe(periods=FORECAST_HORIZON_MONTHS, freq='M')
    
    # Add regressor values for future periods
    # Use last known values or averages for future
    last_values = prophet_df.iloc[-1]
    
    for col in ['geo_risk', 'tariff_risk', 'commodity_pressure', 'weather_risk']:
        if col in prophet_df.columns:
            # For historical periods, use actual values
            future = future.merge(
                prophet_df[['ds', col]], 
                on='ds', 
                how='left'
            )
            # For future periods, use last known value
            future[col] = future[col].fillna(last_values[col])
    
    # Generate forecast
    forecast = model.predict(future)
    
    print(f"Generated forecast for {FORECAST_HORIZON_MONTHS} months ahead")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(FORECAST_HORIZON_MONTHS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Forecast

# COMMAND ----------

if demand_df is not None:
    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title('Oshkosh Defense Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Monthly Obligations (USD)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/tmp/forecast_plot.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/forecast_plot.png')
    
    display(fig1)

# COMMAND ----------

if demand_df is not None:
    # Plot components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/tmp/components_plot.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/components_plot.png')
    
    display(fig2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation with Cross-Validation

# COMMAND ----------

# Cross-validation: need enough history (guide: initial=730 days, horizon=90 days)
# For monthly data, 36 months = 3 years is sufficient for initial training + horizon
if demand_df is not None and len(prophet_df) > 36:
    print("Running cross-validation...")
    
    # Cross-validation (Prophet diagnostics per supply_chain_forecasting_guide)
    cv_results = cross_validation(
        model,
        horizon=CROSS_VALIDATION_HORIZON,
        period=CROSS_VALIDATION_PERIOD,
        initial=CROSS_VALIDATION_INITIAL
    )
    
    # Calculate performance metrics
    metrics = performance_metrics(cv_results)
    
    print("\n=== Cross-Validation Metrics ===")
    print(f"MAPE: {metrics['mape'].mean():.2%}")
    print(f"RMSE: {metrics['rmse'].mean():,.2f}")
    print(f"MAE: {metrics['mae'].mean():,.2f}")
    
    # Log metrics to MLflow
    mlflow.log_metric("cv_mape", metrics['mape'].mean())
    mlflow.log_metric("cv_rmse", metrics['rmse'].mean())
    mlflow.log_metric("cv_mae", metrics['mae'].mean())
    mlflow.log_metric("cv_coverage", metrics['coverage'].mean())

# COMMAND ----------

if demand_df is not None and 'cv_results' in dir():
    # Plot cross-validation results
    from prophet.plot import plot_cross_validation_metric
    
    fig3 = plot_cross_validation_metric(cv_results, metric='mape')
    plt.title('MAPE by Forecast Horizon')
    plt.tight_layout()
    
    plt.savefig('/tmp/cv_mape_plot.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/cv_mape_plot.png')
    
    display(fig3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Forecast Results

# COMMAND ----------

if demand_df is not None:
    # Prepare forecast results for saving (only columns that exist in predict output)
    base_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
    optional = [c for c in ['yearly', 'fiscal_quarterly'] if c in forecast.columns]
    forecast_results = forecast[base_cols + optional].copy()
    col_map = {'ds': 'month', 'yhat': 'forecast_demand_usd', 'yhat_lower': 'forecast_lower',
               'yhat_upper': 'forecast_upper', 'yearly': 'yearly_seasonality',
               'fiscal_quarterly': 'fiscal_quarterly_seasonality'}
    forecast_results.rename(columns=col_map, inplace=True)
    
    # Add metadata
    forecast_results['model_type'] = 'PROPHET'
    forecast_results['model_run_id'] = run_id
    forecast_results['forecast_generated_at'] = datetime.now()
    
    # Convert to Spark and save (Unity Catalog)
    forecast_spark = spark.createDataFrame(forecast_results)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
    forecast_spark.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(FORECAST_OUTPUT_TABLE)
    
    print(f"Saved {len(forecast_results)} forecast records to {FORECAST_OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model in Unity Catalog

# COMMAND ----------

# Register model
if demand_df is not None:
    try:
        # Set registry URI
        mlflow.set_registry_uri("databricks-uc")
        
        # Register model
        model_name = "supply_chain.models.prophet_demand_forecast"
        
        model_version = mlflow.register_model(
            f"runs:/{run_id}/prophet_model",
            model_name
        )
        
        print(f"Registered model: {model_name}")
        print(f"Version: {model_version.version}")
        
        # Set alias
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=model_version.version
        )
        
        print("Set 'champion' alias for production deployment")
        
    except Exception as e:
        print(f"Model registration error (may require Unity Catalog setup): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Summary

# COMMAND ----------

if demand_df is not None:
    print("=== Prophet Demand Forecast Model Summary ===")
    print(f"\nTraining Data:")
    print(f"  - Records: {len(prophet_df)}")
    print(f"  - Date Range: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
    
    print(f"\nModel Configuration:")
    print(f"  - Seasonality Mode: multiplicative")
    print(f"  - Custom Seasonality: fiscal_quarterly")
    print(f"  - Holidays/Events: {len(defense_events)}")
    print(f"  - Exogenous Regressors: geo_risk, tariff_risk, commodity_pressure, weather_risk")
    
    print(f"\nForecast:")
    print(f"  - Horizon: {FORECAST_HORIZON_MONTHS} months")
    print(f"  - Confidence Interval: 95%")
    
    if 'metrics' in dir():
        print(f"\nCross-Validation Metrics:")
        print(f"  - MAPE: {metrics['mape'].mean():.2%}")
        print(f"  - RMSE: {metrics['rmse'].mean():,.2f}")
        print(f"  - MAE: {metrics['mae'].mean():,.2f}")
    
    print(f"\nMLflow Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `02_arima_forecasting` for alternative model comparison
# MAGIC 2. Run `03_model_comparison` to evaluate all models
# MAGIC 3. Proceed to agent tools notebooks