# Databricks notebook source
# MAGIC %md
# MAGIC # ARIMA Demand Forecasting with MLflow
# MAGIC
# MAGIC **Executive summary:** Trains SARIMAX demand forecast with auto-ARIMA and exogenous variables; logs to MLflow and writes gold forecasts. Management: alternative to Prophet for comparison; use for demand planning.
# MAGIC
# MAGIC **Depends on:** `supply_chain.gold.oshkosh_monthly_demand_signals` (run transformation notebooks first).
# MAGIC
# MAGIC This notebook implements demand forecasting using ARIMA/SARIMAX models with:
# MAGIC - Automatic parameter selection (auto-ARIMA)
# MAGIC - Seasonal components for fiscal year patterns
# MAGIC - Exogenous variables support
# MAGIC - MLflow experiment tracking
# MAGIC
# MAGIC **Model**: SARIMAX (Seasonal ARIMA with eXogenous variables)
# MAGIC **Target**: Monthly demand obligations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Install required packages. Use NumPy 1.x (>=1.26.4,<2) for numpy.exceptions and compatibility with pmdarima/statsmodels.
# After this runs: if you see "Core Python package version(s) changed" or "Failed to set environment metadata",
# detach and re-attach the notebook to the cluster (or restart the cluster) then run the notebook again.
%pip install "numpy>=1.26.4,<2" "pandas>=2.0,<3" pmdarima statsmodels mlflow scikit-learn matplotlib

# COMMAND ----------

# Restart Python so the installed NumPy is loaded (required for numpy.exceptions / pmdarima).
try:
    import numpy as _np
    _v = tuple(int(x) for x in getattr(_np, "__version__", "0.0").split(".")[:2])
    if _v < (1, 26) or _v >= (2, 0) or not hasattr(_np, "exceptions"):
        dbutils.library.restartPython()
except Exception:
    dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import mlflow
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import functions as F

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
DEMAND_SIGNALS_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"
FORECAST_OUTPUT_TABLE = f"{CATALOG}.gold.arima_forecasts"
# When DBFS root is disabled, use a workspace path (e.g. /Workspace/Users/<you>/models/arima_demand_forecast)
MODEL_PATH = "/models/arima_demand_forecast"

# MLflow configuration
EXPERIMENT_NAME = "/Shared/supply_chain_platform/experiments/demand_forecasting"

# Forecast parameters
FORECAST_HORIZON_MONTHS = 12
TEST_SIZE_MONTHS = 12  # Hold out last 12 months for testing

# ARIMA parameters
SEASONAL_PERIOD = 12  # Monthly data with yearly seasonality

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow

# COMMAND ----------

# Set up MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load demand signals (Unity Catalog). Use SQL to avoid "Path must be absolute" when table name is mistaken for a path.
try:
    demand_spark = spark.sql(f"SELECT * FROM {DEMAND_SIGNALS_TABLE}")
    demand_df = demand_spark.toPandas()
    print(f"Loaded {len(demand_df)} demand signal records")
except Exception as e:
    print(f"Error loading demand signals: {e}")
    demand_df = None

# COMMAND ----------

if demand_df is not None:
    # Prepare data for ARIMA
    arima_df = demand_df[['month', 'total_obligations_usd', 
                          'geo_risk_index', 'tariff_risk_index',
                          'commodity_cost_pressure', 'weather_disruption_index']].copy()
    
    arima_df['month'] = pd.to_datetime(arima_df['month'])
    arima_df = arima_df.sort_values('month')
    arima_df = arima_df.set_index('month')
    
    # Remove rows with zero demand
    arima_df = arima_df[arima_df['total_obligations_usd'] > 0]
    
    print(f"Prepared {len(arima_df)} records for ARIMA")
    print(f"Date range: {arima_df.index.min()} to {arima_df.index.max()}")
    print(arima_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split

# COMMAND ----------

if demand_df is not None:
    # Split data
    train_size = len(arima_df) - TEST_SIZE_MONTHS
    
    train_df = arima_df.iloc[:train_size]
    test_df = arima_df.iloc[train_size:]
    
    # Target variable
    y_train = train_df['total_obligations_usd']
    y_test = test_df['total_obligations_usd']
    
    # Exogenous variables
    exog_cols = ['geo_risk_index', 'tariff_risk_index', 'commodity_cost_pressure', 'weather_disruption_index']
    X_train = train_df[exog_cols]
    X_test = test_df[exog_cols]
    
    print(f"Training set: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test set: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-ARIMA Parameter Selection

# COMMAND ----------

if demand_df is not None:
    print("Running auto-ARIMA to find optimal parameters...")
    
    with mlflow.start_run(run_name="arima_demand_forecast"):
        # Auto-ARIMA to find best parameters
        auto_model = auto_arima(
            y_train,
            X=X_train,
            seasonal=True,
            m=SEASONAL_PERIOD,  # Monthly seasonality
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            d=None,  # Auto-detect differencing
            D=None,  # Auto-detect seasonal differencing
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic'
        )
        
        # Get best parameters
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        
        print(f"\nBest ARIMA order: {order}")
        print(f"Best seasonal order: {seasonal_order}")
        
        # Log parameters
        mlflow.log_param("arima_order", str(order))
        mlflow.log_param("seasonal_order", str(seasonal_order))
        mlflow.log_param("seasonal_period", SEASONAL_PERIOD)
        mlflow.log_param("training_samples", len(y_train))
        mlflow.log_param("exogenous_variables", str(exog_cols))
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train SARIMAX Model

# COMMAND ----------

if demand_df is not None:
    # Train SARIMAX model with optimal parameters
    print("Training SARIMAX model...")
    
    sarimax_model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    sarimax_results = sarimax_model.fit(disp=False)
    
    print("\n=== Model Summary ===")
    print(sarimax_results.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Forecasts

# COMMAND ----------

if demand_df is not None:
    # In-sample predictions (fitted values)
    fitted_values = sarimax_results.fittedvalues
    
    # Out-of-sample forecast for test period
    forecast_test = sarimax_results.get_forecast(
        steps=len(test_df),
        exog=X_test
    )
    
    forecast_mean = forecast_test.predicted_mean
    forecast_ci = forecast_test.conf_int(alpha=0.05)
    
    print("Test Period Forecast:")
    forecast_results = pd.DataFrame({
        'actual': y_test,
        'forecast': forecast_mean,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1]
    })
    print(forecast_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

if demand_df is not None:
    # Calculate metrics
    mae = mean_absolute_error(y_test, forecast_mean)
    rmse = np.sqrt(mean_squared_error(y_test, forecast_mean))
    mape = mean_absolute_percentage_error(y_test, forecast_mean)
    
    # Calculate coverage (% of actuals within confidence interval)
    coverage = np.mean(
        (y_test >= forecast_ci.iloc[:, 0]) & 
        (y_test <= forecast_ci.iloc[:, 1])
    )
    
    print("=== Model Evaluation Metrics ===")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"Coverage (95% CI): {coverage:.2%}")
    
    # Log metrics to MLflow
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mape", mape)
    mlflow.log_metric("test_coverage", coverage)
    mlflow.log_metric("aic", sarimax_results.aic)
    mlflow.log_metric("bic", sarimax_results.bic)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results

# COMMAND ----------

if demand_df is not None:
    # Plot actual vs forecast
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full time series with forecast
    ax1 = axes[0]
    ax1.plot(y_train.index, y_train, label='Training Data', color='blue')
    ax1.plot(y_test.index, y_test, label='Actual (Test)', color='green')
    ax1.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red', linestyle='--')
    ax1.fill_between(
        forecast_ci.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color='red', alpha=0.2, label='95% CI'
    )
    ax1.set_title('SARIMAX Demand Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Monthly Obligations (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test period detail
    ax2 = axes[1]
    ax2.plot(y_test.index, y_test, 'go-', label='Actual', markersize=8)
    ax2.plot(forecast_mean.index, forecast_mean, 'r^--', label='Forecast', markersize=8)
    ax2.fill_between(
        forecast_ci.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color='red', alpha=0.2
    )
    ax2.set_title('Test Period: Actual vs Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Monthly Obligations (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/arima_forecast_plot.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/arima_forecast_plot.png')
    
    display(fig)

# COMMAND ----------

if demand_df is not None:
    # Residual diagnostics
    fig2 = sarimax_results.plot_diagnostics(figsize=(14, 10))
    plt.tight_layout()
    plt.savefig('/tmp/arima_diagnostics.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/arima_diagnostics.png')
    
    display(fig2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Future Forecast

# COMMAND ----------

if demand_df is not None:
    # Create future exogenous variables (use last known values)
    last_exog = X_test.iloc[-1]
    
    future_exog = pd.DataFrame(
        [last_exog.values] * FORECAST_HORIZON_MONTHS,
        columns=exog_cols,
        index=pd.date_range(
            start=arima_df.index[-1] + pd.DateOffset(months=1),
            periods=FORECAST_HORIZON_MONTHS,
            freq='M'
        )
    )
    
    # Generate future forecast
    future_forecast = sarimax_results.get_forecast(
        steps=FORECAST_HORIZON_MONTHS,
        exog=future_exog
    )
    
    future_mean = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int(alpha=0.05)
    
    print("=== Future Forecast (Next 12 Months) ===")
    future_results = pd.DataFrame({
        'forecast': future_mean,
        'lower_ci': future_ci.iloc[:, 0],
        'upper_ci': future_ci.iloc[:, 1]
    })
    print(future_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Forecast Results

# COMMAND ----------

if demand_df is not None:
    # Combine historical fitted values and future forecast
    all_forecasts = pd.DataFrame({
        'month': list(fitted_values.index) + list(forecast_mean.index) + list(future_mean.index),
        'forecast_demand_usd': list(fitted_values.values) + list(forecast_mean.values) + list(future_mean.values),
        'is_forecast': [False] * len(fitted_values) + [True] * (len(forecast_mean) + len(future_mean))
    })
    
    # Add confidence intervals for forecast periods
    all_forecasts['forecast_lower'] = None
    all_forecasts['forecast_upper'] = None
    
    # Add metadata
    all_forecasts['model_type'] = 'SARIMAX'
    all_forecasts['arima_order'] = str(order)
    all_forecasts['seasonal_order'] = str(seasonal_order)
    all_forecasts['model_run_id'] = run_id
    all_forecasts['forecast_generated_at'] = datetime.now()
    
    # Convert to Spark and save (Unity Catalog)
    forecast_spark = spark.createDataFrame(all_forecasts)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
    forecast_spark.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(FORECAST_OUTPUT_TABLE)
    
    print(f"Saved {len(all_forecasts)} forecast records to {FORECAST_OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison Summary

# COMMAND ----------

if demand_df is not None:
    print("=== ARIMA/SARIMAX Model Summary ===")
    print(f"\nModel Configuration:")
    print(f"  - ARIMA Order (p,d,q): {order}")
    print(f"  - Seasonal Order (P,D,Q,s): {seasonal_order}")
    print(f"  - Exogenous Variables: {exog_cols}")
    
    print(f"\nTraining Data:")
    print(f"  - Records: {len(y_train)}")
    print(f"  - Date Range: {y_train.index.min().date()} to {y_train.index.max().date()}")
    
    print(f"\nTest Performance:")
    print(f"  - MAE: {mae:,.2f}")
    print(f"  - RMSE: {rmse:,.2f}")
    print(f"  - MAPE: {mape:.2%}")
    print(f"  - Coverage: {coverage:.2%}")
    
    print(f"\nModel Fit Statistics:")
    print(f"  - AIC: {sarimax_results.aic:.2f}")
    print(f"  - BIC: {sarimax_results.bic:.2f}")
    
    print(f"\nMLflow Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast Table
# MAGIC Forecasts are written to Unity Catalog by saveAsTable(FORECAST_OUTPUT_TABLE) above.
# MAGIC No manual registration needed. If DBFS root is disabled, do not use CREATE TABLE ... LOCATION '/path'.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run `03_model_comparison` to compare Prophet vs ARIMA
# MAGIC 2. Proceed to agent tools notebooks