# Databricks notebook source
# MAGIC %md
# MAGIC # Random Forest Demand Forecasting with Feature Importance
# MAGIC
# MAGIC **Executive summary:** Implements Random Forest regression for demand forecasting with comprehensive feature engineering and explainability. Unlike Prophet/ARIMA, RF captures complex non-linear relationships between demand and risk signals. Management: use for feature importance insights and scenario modeling.
# MAGIC
# MAGIC **Depends on:** `gold.oshkosh_monthly_demand_signals` (from 01_unified_demand_signals)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `gold.random_forest_forecasts` - Forecast predictions with confidence intervals
# MAGIC - `gold.random_forest_feature_importance` - Feature importance rankings
# MAGIC - MLflow logged model with metrics
# MAGIC
# MAGIC **Key Features:**
# MAGIC - Engineered lag features (1, 3, 6, 12 months)
# MAGIC - Rolling statistics (mean, std, trend)
# MAGIC - Seasonal indicators
# MAGIC - Risk signal interactions
# MAGIC - SHAP values for explainability

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

%pip install scikit-learn shap mlflow pandas numpy matplotlib seaborn

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import shap

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
DEMAND_SIGNALS_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"
FORECAST_OUTPUT_TABLE = f"{CATALOG}.gold.random_forest_forecasts"
FEATURE_IMPORTANCE_TABLE = f"{CATALOG}.gold.random_forest_feature_importance"

# Model parameters
FORECAST_HORIZON = 12  # months ahead
TEST_SIZE = 12  # months for testing
N_ESTIMATORS = 200
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Data

# COMMAND ----------

# Load demand signals
demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
demand_df['month'] = pd.to_datetime(demand_df['month'])
demand_df = demand_df.sort_values('month').reset_index(drop=True)

print(f"✓ Loaded {len(demand_df)} months of data")
print(f"  Date range: {demand_df['month'].min()} to {demand_df['month'].max()}")
print(f"  Columns: {len(demand_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set for Random Forest.
    
    Features include:
    - Lag features (1, 3, 6, 12 months)
    - Rolling statistics (mean, std, min, max)
    - Trend indicators
    - Seasonal features
    - Risk signal interactions
    """
    df = df.copy()
    
    # Target variable
    target_col = 'total_obligations_usd'
    
    # === LAG FEATURES ===
    for lag in [1, 3, 6, 12]:
        df[f'demand_lag_{lag}m'] = df[target_col].shift(lag)
    
    # === ROLLING STATISTICS ===
    for window in [3, 6, 12]:
        df[f'demand_rolling_mean_{window}m'] = df[target_col].shift(1).rolling(window).mean()
        df[f'demand_rolling_std_{window}m'] = df[target_col].shift(1).rolling(window).std()
        df[f'demand_rolling_min_{window}m'] = df[target_col].shift(1).rolling(window).min()
        df[f'demand_rolling_max_{window}m'] = df[target_col].shift(1).rolling(window).max()
    
    # === TREND FEATURES ===
    df['demand_trend_3m'] = df[target_col].shift(1) - df[target_col].shift(3)
    df['demand_trend_6m'] = df[target_col].shift(1) - df[target_col].shift(6)
    df['demand_pct_change_1m'] = df[target_col].pct_change(1)
    df['demand_pct_change_3m'] = df[target_col].pct_change(3)
    
    # === SEASONAL FEATURES ===
    df['month_of_year'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df['is_q4'] = (df['quarter'] == 4).astype(int)  # Fiscal year end
    df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
    
    # === RISK SIGNAL INTERACTIONS ===
    if 'geo_risk_index' in df.columns and 'tariff_risk_index' in df.columns:
        df['combined_risk_interaction'] = df['geo_risk_index'] * df['tariff_risk_index']
        df['geo_risk_lag_1m'] = df['geo_risk_index'].shift(1)
        df['tariff_risk_lag_1m'] = df['tariff_risk_index'].shift(1)
    
    if 'commodity_cost_pressure' in df.columns:
        df['commodity_lag_1m'] = df['commodity_cost_pressure'].shift(1)
        df['commodity_trend_3m'] = df['commodity_cost_pressure'].shift(1) - df['commodity_cost_pressure'].shift(3)
    
    if 'weather_disruption_index' in df.columns:
        df['weather_lag_1m'] = df['weather_disruption_index'].shift(1)
    
    # === TIME FEATURES ===
    df['months_since_start'] = (df['month'] - df['month'].min()).dt.days / 30.44
    
    return df

# Engineer features
df_engineered = engineer_features(demand_df)

# Drop rows with NaN (due to lag/rolling features)
df_clean = df_engineered.dropna()

print(f"✓ Feature engineering complete")
print(f"  Original features: {len(demand_df.columns)}")
print(f"  Engineered features: {len(df_engineered.columns)}")
print(f"  Clean records: {len(df_clean)} (after removing NaN)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Features and Split Data

# COMMAND ----------

# Define feature columns (exclude target, date, and identifiers)
exclude_cols = ['month', 'total_obligations_usd', 'prime_obligations_usd', 'subaward_obligations_usd']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols and df_clean[col].dtype in ['float64', 'int64']]

print(f"✓ Selected {len(feature_cols)} features:")
for col in sorted(feature_cols):
    print(f"  - {col}")

# Prepare X and y
X = df_clean[feature_cols].values
y = df_clean['total_obligations_usd'].values
dates = df_clean['month'].values

# Time series split (train on past, test on recent)
train_size = len(X) - TEST_SIZE
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
dates_train, dates_test = dates[:train_size], dates[train_size:]

print(f"\n✓ Data split:")
print(f"  Training: {len(X_train)} months ({dates_train[0]} to {dates_train[-1]})")
print(f"  Testing: {len(X_test)} months ({dates_test[0]} to {dates_test[-1]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Random Forest Model

# COMMAND ----------

# Start MLflow run
mlflow.set_experiment("/SupplyChain/RandomForestForecasting")

with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    
    # Log parameters
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
    mlflow.log_param("forecast_horizon", FORECAST_HORIZON)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("n_features", len(feature_cols))
    
    # Train model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    # Log metrics
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_mape", train_mape)
    mlflow.log_metric("test_mape", test_mape)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    print("\n=== MODEL PERFORMANCE ===")
    print(f"Training Set:")
    print(f"  MAE:  ${train_mae:,.0f}")
    print(f"  RMSE: ${train_rmse:,.0f}")
    print(f"  MAPE: {train_mape:.2f}%")
    print(f"\nTest Set:")
    print(f"  MAE:  ${test_mae:,.0f}")
    print(f"  RMSE: ${test_rmse:,.0f}")
    print(f"  MAPE: {test_mape:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 FEATURE IMPORTANCES ===")
for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:40s} {row['importance']:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_n = 20
sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature', palette='viridis')
plt.title(f'Top {top_n} Feature Importances - Random Forest Demand Forecasting', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()

# Save plot
importance_plot_path = "/tmp/rf_feature_importance.png"
plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
mlflow.log_artifact(importance_plot_path)
print(f"\n✓ Feature importance plot saved")

# Save to Delta table
feature_importance['model_type'] = 'RandomForest'
feature_importance['training_date'] = datetime.now()
feature_importance['rank'] = range(1, len(feature_importance) + 1)

spark.createDataFrame(feature_importance) \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(FEATURE_IMPORTANCE_TABLE)

print(f"✓ Feature importance saved to {FEATURE_IMPORTANCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP Values for Explainability

# COMMAND ----------

print("Calculating SHAP values (this may take a few minutes)...")

# Use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.tight_layout()

shap_plot_path = "/tmp/rf_shap_summary.png"
plt.savefig(shap_plot_path, dpi=150, bbox_inches='tight')
mlflow.log_artifact(shap_plot_path)
print("✓ SHAP summary plot saved")

# Feature importance from SHAP
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("\n=== TOP 10 SHAP FEATURE IMPORTANCES ===")
for idx, row in shap_importance.head(10).iterrows():
    print(f"{row['feature']:40s} {row['shap_importance']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Future Forecasts

# COMMAND ----------

def forecast_future(model, df_historical, feature_cols, n_months=12):
    """
    Generate future forecasts by iteratively predicting and updating features.
    """
    df_forecast = df_historical.copy()
    last_date = df_forecast['month'].max()
    
    forecasts = []
    
    for i in range(1, n_months + 1):
        # Create next month
        next_month = last_date + pd.DateOffset(months=i)
        
        # Re-engineer features with updated data
        df_temp = engineer_features(df_forecast)
        df_temp = df_temp.dropna()
        
        # Get features for last row
        if len(df_temp) > 0:
            X_next = df_temp[feature_cols].iloc[-1:].values
            
            # Predict
            y_pred = model.predict(X_next)[0]
            
            # Create forecast row
            forecast_row = df_forecast.iloc[-1:].copy()
            forecast_row['month'] = next_month
            forecast_row['total_obligations_usd'] = y_pred
            
            # Append to historical data for next iteration
            df_forecast = pd.concat([df_forecast, forecast_row], ignore_index=True)
            
            forecasts.append({
                'month': next_month,
                'forecast_demand_usd': y_pred
            })
    
    return pd.DataFrame(forecasts)

# Generate forecasts
print(f"Generating {FORECAST_HORIZON}-month forecast...")
future_forecasts = forecast_future(rf_model, df_clean, feature_cols, n_months=FORECAST_HORIZON)

print(f"✓ Generated {len(future_forecasts)} months of forecasts")
print(f"  Forecast range: {future_forecasts['month'].min()} to {future_forecasts['month'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Prediction Intervals

# COMMAND ----------

# Use quantile regression forests approach (estimate from tree predictions)
def calculate_prediction_intervals(model, X, confidence=0.95):
    """
    Calculate prediction intervals using individual tree predictions.
    """
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    
    # Calculate percentiles
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)
    
    return lower_bound, upper_bound

# Calculate intervals for test set
y_test_lower, y_test_upper = calculate_prediction_intervals(rf_model, X_test)

# Add to test predictions
test_results = pd.DataFrame({
    'month': dates_test,
    'actual_demand_usd': y_test,
    'forecast_demand_usd': y_test_pred,
    'forecast_lower': y_test_lower,
    'forecast_upper': y_test_upper,
    'is_actual': True
})

# Add future forecasts (no actuals, no intervals for simplicity)
future_forecasts['forecast_lower'] = future_forecasts['forecast_demand_usd'] * 0.85
future_forecasts['forecast_upper'] = future_forecasts['forecast_demand_usd'] * 1.15
future_forecasts['actual_demand_usd'] = None
future_forecasts['is_actual'] = False

# Combine
all_forecasts = pd.concat([test_results, future_forecasts], ignore_index=True)

print(f"✓ Prediction intervals calculated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Forecasts

# COMMAND ----------

# Plot actual vs forecast
plt.figure(figsize=(16, 8))

# Historical actual
plt.plot(dates_train, y_train, 'o-', color='blue', label='Training Data', alpha=0.6, markersize=4)

# Test actual vs predicted
plt.plot(dates_test, y_test, 'o-', color='green', label='Test Actual', markersize=6)
plt.plot(dates_test, y_test_pred, 's-', color='orange', label='Test Predicted', markersize=6)
plt.fill_between(dates_test, y_test_lower, y_test_upper, color='orange', alpha=0.2, label='95% Prediction Interval')

# Future forecast
future_dates = future_forecasts['month'].values
future_preds = future_forecasts['forecast_demand_usd'].values
future_lower = future_forecasts['forecast_lower'].values
future_upper = future_forecasts['forecast_upper'].values

plt.plot(future_dates, future_preds, 's--', color='red', label='Future Forecast', markersize=6)
plt.fill_between(future_dates, future_lower, future_upper, color='red', alpha=0.2)

plt.title('Random Forest Demand Forecast - Oshkosh Defense Contracts', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Demand (USD)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

forecast_plot_path = "/tmp/rf_forecast_plot.png"
plt.savefig(forecast_plot_path, dpi=150, bbox_inches='tight')
mlflow.log_artifact(forecast_plot_path)
print("✓ Forecast plot saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Forecasts to Delta

# COMMAND ----------

# Save to Delta table
spark.createDataFrame(all_forecasts) \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(FORECAST_OUTPUT_TABLE)

print(f"✓ Forecasts saved to {FORECAST_OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("RANDOM FOREST FORECASTING SUMMARY")
print("=" * 60)
print(f"Model: Random Forest Regressor")
print(f"  - Estimators: {N_ESTIMATORS}")
print(f"  - Max Depth: {MAX_DEPTH}")
print(f"  - Features: {len(feature_cols)}")
print(f"\nPerformance (Test Set):")
print(f"  - MAE:  ${test_mae:,.0f}")
print(f"  - RMSE: ${test_rmse:,.0f}")
print(f"  - MAPE: {test_mape:.2f}%")
print(f"\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
print(f"\nForecasts Generated: {len(future_forecasts)} months")
print(f"  Range: {future_forecasts['month'].min().strftime('%Y-%m')} to {future_forecasts['month'].max().strftime('%Y-%m')}")
print(f"\nOutputs:")
print(f"  - {FORECAST_OUTPUT_TABLE}")
print(f"  - {FEATURE_IMPORTANCE_TABLE}")
print("=" * 60)

# Display sample forecasts
display(all_forecasts.tail(15))
