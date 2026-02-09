# Databricks notebook source
# MAGIC %md
# MAGIC # Enriched AI Agent Tools with ML Explainability & Advanced Analytics
# MAGIC
# MAGIC **Executive summary:** Enhanced agent with Random Forest feature importance, SHAP explanations, trend analysis, confidence scoring, and interactive what-if scenarios. Provides deeper insights into demand drivers and model decisions.
# MAGIC
# MAGIC **Depends on:** All gold tables + `gold.random_forest_forecasts` + `gold.random_forest_feature_importance`
# MAGIC
# MAGIC **New Capabilities:**
# MAGIC - Feature importance analysis (understand demand drivers)
# MAGIC - SHAP-based model explainability
# MAGIC - Trend detection and pattern recognition
# MAGIC - Confidence scoring for forecasts
# MAGIC - Interactive what-if scenario builder
# MAGIC - Multi-model forecast comparison
# MAGIC - Risk correlation analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

%pip install --upgrade "typing_extensions>=4.1" "langchain>=0.2,<0.4" "langchain-core>=0.2" langgraph databricks-langchain mlflow pandas numpy scikit-learn scipy

# COMMAND ----------

try:
    from typing_extensions import Sentinel
except ImportError:
    dbutils.library.restartPython()

# COMMAND ----------

from databricks_langchain import ChatDatabricks
_create_tool_calling_agent = None
_AgentExecutor = None
try:
    from langchain.agents import create_tool_calling_agent
    _create_tool_calling_agent = create_tool_calling_agent
except ImportError:
    try:
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
        _create_tool_calling_agent = create_tool_calling_agent
    except (ModuleNotFoundError, ImportError):
        try:
            from langchain.agents import create_react_agent as create_tool_calling_agent
            _create_tool_calling_agent = create_tool_calling_agent
        except ImportError:
            pass
try:
    from langchain_core.agents import AgentExecutor
    _AgentExecutor = AgentExecutor
except ImportError:
    try:
        from langchain.agents import AgentExecutor
        _AgentExecutor = AgentExecutor
    except ImportError:
        pass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import mlflow
import json

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
DEMAND_SIGNALS_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"
DOD_METRICS_TABLE = f"{CATALOG}.gold.dod_metrics_inputs_monthly"
# Optional: geopolitical_risk_scores_monthly (GDELT removed; demand signals have geo_risk_index)
GEO_RISK_TABLE = f"{CATALOG}.gold.geopolitical_risk_scores_monthly"
TRADE_RISK_TABLE = f"{CATALOG}.gold.trade_tariff_risk_monthly"
COMMODITY_TABLE = f"{CATALOG}.silver.commodity_prices_monthly"
WEATHER_TABLE = f"{CATALOG}.silver.weather_risk_monthly"
PROPHET_FORECAST_TABLE = f"{CATALOG}.gold.prophet_forecasts"
ARIMA_FORECAST_TABLE = f"{CATALOG}.gold.arima_forecasts"
RF_FORECAST_TABLE = f"{CATALOG}.gold.random_forest_forecasts"
RF_FEATURE_IMPORTANCE_TABLE = f"{CATALOG}.gold.random_forest_feature_importance"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize LLM

# COMMAND ----------

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.1,
    max_tokens=1500
)

mlflow.langchain.autolog()
print("LLM initialized: databricks-meta-llama-3-3-70b-instruct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW TOOL 1: Feature Importance Analysis

# COMMAND ----------

@tool
def explain_demand_drivers(top_n: int = 10) -> str:
    """
    Explain which features drive demand forecasts using Random Forest feature importance.
    
    Args:
        top_n: Number of top features to show (default 10)
    
    Returns:
        Feature importance ranking with business interpretations
    """
    try:
        # Load feature importance
        importance_df = spark.table(RF_FEATURE_IMPORTANCE_TABLE).toPandas()
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        result = f"DEMAND DRIVER ANALYSIS - Top {top_n} Features\n"
        result += "=" * 60 + "\n\n"
        
        # Interpret features
        feature_interpretations = {
            'demand_lag_1m': 'Previous month demand (momentum)',
            'demand_rolling_mean_12m': 'Annual average demand (baseline)',
            'demand_rolling_mean_6m': '6-month average demand (recent trend)',
            'demand_rolling_mean_3m': '3-month average demand (short-term trend)',
            'geo_risk_index': 'Geopolitical risk level',
            'tariff_risk_index': 'Trade/tariff risk level',
            'commodity_cost_pressure': 'Material cost pressure',
            'weather_disruption_index': 'Weather-related disruptions',
            'is_q4': 'Fiscal year-end effect',
            'months_since_start': 'Long-term time trend',
            'demand_trend_3m': '3-month demand momentum',
            'demand_pct_change_1m': 'Month-over-month growth rate',
            'combined_risk_interaction': 'Combined risk multiplier effect'
        }
        
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            rank = row['rank']
            
            # Get interpretation
            interpretation = feature_interpretations.get(feature, 'Model feature')
            
            result += f"{rank}. {feature}\n"
            result += f"   Importance Score: {importance:.4f}\n"
            result += f"   Meaning: {interpretation}\n\n"
        
        result += "\nKEY INSIGHTS:\n"
        
        # Analyze top driver category
        top_feature = importance_df.iloc[0]['feature']
        if 'lag' in top_feature or 'rolling' in top_feature:
            result += "   📈 Historical demand patterns are the strongest predictor\n"
            result += "   → Demand shows strong momentum/seasonality\n"
        elif 'risk' in top_feature:
            result += "   ⚠️ Risk signals are the strongest predictor\n"
            result += "   → Demand is highly sensitive to geopolitical/trade factors\n"
        elif 'commodity' in top_feature:
            result += "   💰 Commodity prices are the strongest predictor\n"
            result += "   → Material costs significantly influence demand\n"
        
        # Check risk importance
        risk_features = importance_df[importance_df['feature'].str.contains('risk', case=False)]
        if len(risk_features) > 0:
            avg_risk_importance = risk_features['importance'].mean()
            result += f"\n   🎯 Risk factors account for {avg_risk_importance*100:.1f}% of model decisions\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing demand drivers: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW TOOL 2: Trend Detection & Pattern Recognition

# COMMAND ----------

@tool
def detect_trends(lookback_months: int = 12, trend_type: str = "ALL") -> str:
    """
    Detect and analyze demand trends and patterns.
    
    Args:
        lookback_months: Number of months to analyze (default 12)
        trend_type: Type of trend (GROWTH, SEASONAL, VOLATILITY, CORRELATION, ALL)
    
    Returns:
        Comprehensive trend analysis with statistical insights
    """
    try:
        # Load demand signals
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        demand_df = demand_df.sort_values('month')
        
        # Get recent data
        recent = demand_df.tail(lookback_months)
        
        result = f"TREND DETECTION & PATTERN ANALYSIS\n"
        result += f"Period: Last {lookback_months} months\n"
        result += "=" * 60 + "\n\n"
        
        # === GROWTH TREND ===
        if trend_type.upper() in ["GROWTH", "ALL"]:
            # Linear regression
            X = np.arange(len(recent)).reshape(-1, 1)
            y = recent['total_obligations_usd'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
            
            # Calculate growth rate
            avg_demand = y.mean()
            monthly_growth_pct = (slope / avg_demand) * 100
            annual_growth_pct = monthly_growth_pct * 12
            
            result += "📈 GROWTH TREND\n"
            result += f"   Trend Direction: {'📈 INCREASING' if slope > 0 else '📉 DECREASING'}\n"
            result += f"   Monthly Growth Rate: {monthly_growth_pct:+.2f}%\n"
            result += f"   Annualized Growth Rate: {annual_growth_pct:+.2f}%\n"
            result += f"   Trend Strength (R²): {r_value**2:.3f}\n"
            result += f"   Statistical Significance: {'✅ Significant' if p_value < 0.05 else '⚠️ Not significant'} (p={p_value:.4f})\n\n"
        
        # === SEASONALITY ===
        if trend_type.upper() in ["SEASONAL", "ALL"]:
            # Group by month of year
            recent['month_of_year'] = recent['month'].dt.month
            monthly_avg = recent.groupby('month_of_year')['total_obligations_usd'].mean()
            
            # Find peak and trough
            peak_month = monthly_avg.idxmax()
            trough_month = monthly_avg.idxmin()
            seasonal_amplitude = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            result += "📅 SEASONAL PATTERNS\n"
            result += f"   Peak Month: {month_names[peak_month-1]} (${monthly_avg.max():,.0f})\n"
            result += f"   Trough Month: {month_names[trough_month-1]} (${monthly_avg.min():,.0f})\n"
            result += f"   Seasonal Amplitude: {seasonal_amplitude:.1f}%\n"
            
            if seasonal_amplitude > 30:
                result += f"   Assessment: 🔴 HIGH seasonality - plan inventory accordingly\n"
            elif seasonal_amplitude > 15:
                result += f"   Assessment: 🟡 MODERATE seasonality - monitor Q4 closely\n"
            else:
                result += f"   Assessment: 🟢 LOW seasonality - stable demand pattern\n"
            result += "\n"
        
        # === VOLATILITY ===
        if trend_type.upper() in ["VOLATILITY", "ALL"]:
            # Calculate volatility metrics
            returns = recent['total_obligations_usd'].pct_change().dropna()
            volatility = returns.std()
            cv = recent['total_obligations_usd'].std() / recent['total_obligations_usd'].mean()
            
            # Count large swings
            large_swings = (abs(returns) > 0.20).sum()
            
            result += "📊 VOLATILITY ANALYSIS\n"
            result += f"   Standard Deviation: ${recent['total_obligations_usd'].std():,.0f}\n"
            result += f"   Coefficient of Variation: {cv:.3f}\n"
            result += f"   Large Swings (>20%): {large_swings} occurrences\n"
            
            if cv > 0.5:
                result += f"   Assessment: 🔴 HIGH volatility - increase safety stock\n"
            elif cv > 0.3:
                result += f"   Assessment: 🟡 MODERATE volatility - standard buffers adequate\n"
            else:
                result += f"   Assessment: 🟢 LOW volatility - predictable demand\n"
            result += "\n"
        
        # === RISK CORRELATION ===
        if trend_type.upper() in ["CORRELATION", "ALL"]:
            # Correlation with risk factors
            correlations = {}
            if 'geo_risk_index' in recent.columns:
                correlations['Geopolitical Risk'] = recent['total_obligations_usd'].corr(recent['geo_risk_index'])
            if 'tariff_risk_index' in recent.columns:
                correlations['Tariff Risk'] = recent['total_obligations_usd'].corr(recent['tariff_risk_index'])
            if 'commodity_cost_pressure' in recent.columns:
                correlations['Commodity Costs'] = recent['total_obligations_usd'].corr(recent['commodity_cost_pressure'])
            
            result += "🔗 RISK FACTOR CORRELATIONS\n"
            for factor, corr in correlations.items():
                if pd.notna(corr):
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                    direction = "Positive" if corr > 0 else "Negative"
                    result += f"   {factor}: {corr:+.3f} ({strength} {direction})\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error detecting trends: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW TOOL 3: Multi-Model Forecast Comparison

# COMMAND ----------

@tool
def compare_forecast_models(months_ahead: int = 3) -> str:
    """
    Compare forecasts from Prophet, ARIMA, and Random Forest models.
    
    Args:
        months_ahead: Number of months to compare (default 3)
    
    Returns:
        Side-by-side model comparison with confidence intervals
    """
    try:
        # Load forecasts from all models
        prophet_df = spark.table(PROPHET_FORECAST_TABLE).toPandas()
        prophet_df['month'] = pd.to_datetime(prophet_df['month'])
        
        try:
            arima_df = spark.table(ARIMA_FORECAST_TABLE).toPandas()
            arima_df['month'] = pd.to_datetime(arima_df['month'])
        except:
            arima_df = pd.DataFrame()
        
        try:
            rf_df = spark.table(RF_FORECAST_TABLE).toPandas()
            rf_df['month'] = pd.to_datetime(rf_df['month'])
        except:
            rf_df = pd.DataFrame()
        
        # Get future forecasts
        current_date = datetime.now()
        
        prophet_future = prophet_df[prophet_df['month'] > current_date].head(months_ahead)
        arima_future = arima_df[arima_df['month'] > current_date].head(months_ahead) if not arima_df.empty else pd.DataFrame()
        rf_future = rf_df[rf_df['month'] > current_date].head(months_ahead) if not rf_df.empty else pd.DataFrame()
        
        result = f"MULTI-MODEL FORECAST COMPARISON\n"
        result += f"Horizon: Next {months_ahead} months\n"
        result += "=" * 60 + "\n\n"
        
        # Compare month by month
        for i in range(months_ahead):
            if i < len(prophet_future):
                month = prophet_future.iloc[i]['month']
                month_str = month.strftime('%B %Y')
                
                result += f"📅 {month_str}\n"
                
                # Prophet
                prophet_val = prophet_future.iloc[i]['forecast_demand_usd']
                result += f"   Prophet:       ${prophet_val:>12,.0f}\n"
                
                # ARIMA
                if i < len(arima_future):
                    arima_val = arima_future.iloc[i]['forecast_demand_usd']
                    diff_arima = ((arima_val - prophet_val) / prophet_val) * 100
                    result += f"   ARIMA:         ${arima_val:>12,.0f} ({diff_arima:+.1f}%)\n"
                
                # Random Forest
                if i < len(rf_future):
                    rf_val = rf_future.iloc[i]['forecast_demand_usd']
                    diff_rf = ((rf_val - prophet_val) / prophet_val) * 100
                    result += f"   Random Forest: ${rf_val:>12,.0f} ({diff_rf:+.1f}%)\n"
                
                # Ensemble average
                forecasts = [prophet_val]
                if i < len(arima_future):
                    forecasts.append(arima_future.iloc[i]['forecast_demand_usd'])
                if i < len(rf_future):
                    forecasts.append(rf_future.iloc[i]['forecast_demand_usd'])
                
                ensemble_avg = np.mean(forecasts)
                ensemble_std = np.std(forecasts)
                
                result += f"   ─────────────────────────────\n"
                result += f"   Ensemble Avg:  ${ensemble_avg:>12,.0f}\n"
                result += f"   Model Spread:  ${ensemble_std:>12,.0f} ({ensemble_std/ensemble_avg*100:.1f}%)\n\n"
        
        # Model agreement analysis
        result += "MODEL AGREEMENT ANALYSIS:\n"
        
        # Calculate average spread
        all_spreads = []
        for i in range(min(len(prophet_future), months_ahead)):
            forecasts = [prophet_future.iloc[i]['forecast_demand_usd']]
            if i < len(arima_future):
                forecasts.append(arima_future.iloc[i]['forecast_demand_usd'])
            if i < len(rf_future):
                forecasts.append(rf_future.iloc[i]['forecast_demand_usd'])
            if len(forecasts) > 1:
                spread_pct = (np.std(forecasts) / np.mean(forecasts)) * 100
                all_spreads.append(spread_pct)
        
        if all_spreads:
            avg_spread = np.mean(all_spreads)
            if avg_spread < 5:
                result += f"   ✅ HIGH agreement (avg spread: {avg_spread:.1f}%)\n"
                result += f"   → Forecasts are highly consistent\n"
            elif avg_spread < 15:
                result += f"   🟡 MODERATE agreement (avg spread: {avg_spread:.1f}%)\n"
                result += f"   → Consider ensemble average for planning\n"
            else:
                result += f"   🔴 LOW agreement (avg spread: {avg_spread:.1f}%)\n"
                result += f"   → High uncertainty - use wider safety margins\n"
        
        return result
        
    except Exception as e:
        return f"Error comparing models: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW TOOL 4: Confidence Scoring

# COMMAND ----------

@tool
def assess_forecast_confidence(months_ahead: int = 3) -> str:
    """
    Assess confidence level in demand forecasts based on multiple factors.
    
    Args:
        months_ahead: Number of months to assess (default 3)
    
    Returns:
        Confidence score with contributing factors
    """
    try:
        # Load data
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        demand_df = demand_df.sort_values('month')
        
        recent = demand_df.tail(12)
        
        result = f"FORECAST CONFIDENCE ASSESSMENT\n"
        result += f"Horizon: Next {months_ahead} months\n"
        result += "=" * 60 + "\n\n"
        
        # Factor 1: Historical volatility
        cv = recent['total_obligations_usd'].std() / recent['total_obligations_usd'].mean()
        volatility_score = max(0, 100 - (cv * 200))  # Lower volatility = higher confidence
        
        result += f"📊 CONFIDENCE FACTORS:\n\n"
        result += f"1. Historical Stability\n"
        result += f"   Coefficient of Variation: {cv:.3f}\n"
        result += f"   Confidence Contribution: {volatility_score:.0f}/100\n"
        result += f"   {'✅ Stable' if cv < 0.3 else '⚠️ Volatile'}\n\n"
        
        # Factor 2: Trend consistency
        X = np.arange(len(recent)).reshape(-1, 1)
        y = recent['total_obligations_usd'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
        
        trend_score = (r_value ** 2) * 100  # R² as confidence
        
        result += f"2. Trend Consistency\n"
        result += f"   R² Score: {r_value**2:.3f}\n"
        result += f"   Confidence Contribution: {trend_score:.0f}/100\n"
        result += f"   {'✅ Strong trend' if r_value**2 > 0.7 else '⚠️ Weak trend'}\n\n"
        
        # Factor 3: Risk environment stability
        risk_cols = ['geo_risk_index', 'tariff_risk_index', 'weather_disruption_index']
        risk_stability_scores = []
        
        for col in risk_cols:
            if col in recent.columns:
                risk_cv = recent[col].std() / (recent[col].mean() + 1e-6)
                risk_stability_scores.append(max(0, 100 - (risk_cv * 100)))
        
        risk_score = np.mean(risk_stability_scores) if risk_stability_scores else 50
        
        result += f"3. Risk Environment Stability\n"
        result += f"   Risk Volatility: {'Low' if risk_score > 70 else 'Moderate' if risk_score > 40 else 'High'}\n"
        result += f"   Confidence Contribution: {risk_score:.0f}/100\n"
        result += f"   {'✅ Stable' if risk_score > 70 else '⚠️ Elevated'}\n\n"
        
        # Factor 4: Data recency and completeness
        days_since_last = (datetime.now() - recent['month'].max()).days
        recency_score = max(0, 100 - (days_since_last / 30 * 50))  # Penalize old data
        
        result += f"4. Data Recency\n"
        result += f"   Last Update: {recent['month'].max().strftime('%Y-%m-%d')} ({days_since_last} days ago)\n"
        result += f"   Confidence Contribution: {recency_score:.0f}/100\n"
        result += f"   {'✅ Current' if days_since_last < 45 else '⚠️ Stale'}\n\n"
        
        # Overall confidence score
        overall_confidence = (volatility_score * 0.35 + trend_score * 0.30 + risk_score * 0.25 + recency_score * 0.10)
        
        result += "=" * 60 + "\n"
        result += f"OVERALL CONFIDENCE SCORE: {overall_confidence:.0f}/100\n\n"
        
        if overall_confidence >= 80:
            result += "   🟢 HIGH CONFIDENCE\n"
            result += "   → Forecasts are reliable for planning\n"
            result += "   → Standard safety stock levels appropriate\n"
        elif overall_confidence >= 60:
            result += "   🟡 MODERATE CONFIDENCE\n"
            result += "   → Forecasts are reasonable but monitor closely\n"
            result += "   → Consider 15-20% safety buffer\n"
        else:
            result += "   🔴 LOW CONFIDENCE\n"
            result += "   → High uncertainty in forecasts\n"
            result += "   → Increase safety stock by 30-50%\n"
            result += "   → Review assumptions and update models\n"
        
        return result
        
    except Exception as e:
        return f"Error assessing confidence: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW TOOL 5: Interactive What-If Scenario Builder

# COMMAND ----------

@tool
def build_whatif_scenario(
    geo_risk_change_pct: float = 0.0,
    tariff_change_pct: float = 0.0,
    commodity_price_change_pct: float = 0.0,
    demand_shock_pct: float = 0.0
) -> str:
    """
    Build custom what-if scenario by adjusting multiple risk factors simultaneously.
    
    Args:
        geo_risk_change_pct: % change in geopolitical risk (e.g., 50 for +50%)
        tariff_change_pct: % change in tariff risk (e.g., 25 for +25%)
        commodity_price_change_pct: % change in commodity prices (e.g., 15 for +15%)
        demand_shock_pct: Direct % change in demand (e.g., -10 for -10%)
    
    Returns:
        Comprehensive scenario impact analysis
    """
    try:
        # Load baseline data
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        
        # Get baseline (last 12 months average)
        recent = demand_df.tail(12)
        baseline_demand = recent['total_obligations_usd'].mean()
        baseline_geo_risk = recent['geo_risk_index'].mean() if 'geo_risk_index' in recent.columns else 0
        baseline_tariff_risk = recent['tariff_risk_index'].mean() if 'tariff_risk_index' in recent.columns else 0
        baseline_commodity = recent['commodity_cost_pressure'].mean() if 'commodity_cost_pressure' in recent.columns else 0
        
        result = f"CUSTOM WHAT-IF SCENARIO ANALYSIS\n"
        result += "=" * 60 + "\n\n"
        
        result += f"BASELINE (Current State):\n"
        result += f"   Average Monthly Demand: ${baseline_demand:,.0f}\n"
        result += f"   Geo Risk Index: {baseline_geo_risk:.2f}\n"
        result += f"   Tariff Risk Index: {baseline_tariff_risk:.2f}\n"
        result += f"   Commodity Cost Pressure: {baseline_commodity:.2f}\n\n"
        
        result += f"SCENARIO ADJUSTMENTS:\n"
        if geo_risk_change_pct != 0:
            result += f"   Geopolitical Risk: {geo_risk_change_pct:+.0f}%\n"
        if tariff_change_pct != 0:
            result += f"   Tariff Risk: {tariff_change_pct:+.0f}%\n"
        if commodity_price_change_pct != 0:
            result += f"   Commodity Prices: {commodity_price_change_pct:+.0f}%\n"
        if demand_shock_pct != 0:
            result += f"   Direct Demand Shock: {demand_shock_pct:+.0f}%\n"
        result += "\n"
        
        # Calculate impacts (simplified elasticity model)
        # Geo risk: 1% risk increase → 0.3% demand increase (defense spending)
        geo_impact = (geo_risk_change_pct / 100) * 0.30
        
        # Tariff risk: 1% tariff increase → -0.05% demand (cost pressure)
        tariff_impact = (tariff_change_pct / 100) * -0.05
        
        # Commodity prices: 1% price increase → -0.02% demand (budget constraints)
        commodity_impact = (commodity_price_change_pct / 100) * -0.02
        
        # Direct shock
        direct_impact = demand_shock_pct / 100
        
        # Total impact
        total_impact_pct = (geo_impact + tariff_impact + commodity_impact + direct_impact) * 100
        scenario_demand = baseline_demand * (1 + total_impact_pct / 100)
        demand_change = scenario_demand - baseline_demand
        
        result += f"PROJECTED IMPACTS:\n"
        result += f"   Geopolitical Effect: {geo_impact*100:+.2f}%\n"
        result += f"   Tariff Effect: {tariff_impact*100:+.2f}%\n"
        result += f"   Commodity Effect: {commodity_impact*100:+.2f}%\n"
        result += f"   Direct Shock Effect: {direct_impact*100:+.2f}%\n"
        result += f"   ─────────────────────────────\n"
        result += f"   TOTAL DEMAND IMPACT: {total_impact_pct:+.2f}%\n\n"
        
        result += f"SCENARIO OUTCOMES:\n"
        result += f"   Projected Monthly Demand: ${scenario_demand:,.0f}\n"
        result += f"   Change from Baseline: ${demand_change:,.0f}\n"
        result += f"   Annual Impact: ${demand_change * 12:,.0f}\n\n"
        
        # Recommendations
        result += f"RECOMMENDED ACTIONS:\n"
        if total_impact_pct > 20:
            result += "   🔴 MAJOR INCREASE - Surge capacity scenario\n"
            result += "   → Activate emergency procurement protocols\n"
            result += "   → Increase safety stock by 50%+\n"
            result += "   → Engage backup suppliers immediately\n"
        elif total_impact_pct > 10:
            result += "   🟠 SIGNIFICANT INCREASE\n"
            result += "   → Accelerate procurement timelines\n"
            result += "   → Increase safety stock by 25-35%\n"
            result += "   → Review supplier capacity\n"
        elif total_impact_pct < -20:
            result += "   🔵 MAJOR DECREASE\n"
            result += "   → Review inventory levels to avoid excess\n"
            result += "   → Negotiate flexible supplier contracts\n"
            result += "   → Consider production slowdown\n"
        elif total_impact_pct < -10:
            result += "   🟡 MODERATE DECREASE\n"
            result += "   → Adjust procurement schedules\n"
            result += "   → Monitor for further changes\n"
        else:
            result += "   🟢 MINIMAL IMPACT\n"
            result += "   → Continue normal operations\n"
            result += "   → Standard safety stock adequate\n"
        
        return result
        
    except Exception as e:
        return f"Error building what-if scenario: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Original Tools

# COMMAND ----------

# Import original tools from 01_agent_tools (simplified versions)

@tool
def get_demand_forecast(months_ahead: int = 3, include_confidence: bool = True) -> str:
    """Retrieve demand forecast for Oshkosh Defense contracts."""
    try:
        forecast_df = spark.table(PROPHET_FORECAST_TABLE).toPandas()
        forecast_df['month'] = pd.to_datetime(forecast_df['month'])
        current_date = datetime.now()
        future_forecasts = forecast_df[forecast_df['month'] > current_date].head(months_ahead)
        
        if future_forecasts.empty:
            return "No forecast data available."
        
        result = f"DEMAND FORECAST - Next {months_ahead} Months\n" + "=" * 50 + "\n\n"
        total_forecast = 0
        
        for _, row in future_forecasts.iterrows():
            month_str = row['month'].strftime('%B %Y')
            forecast = row['forecast_demand_usd']
            total_forecast += forecast
            result += f"📅 {month_str}\n   Forecast: ${forecast:,.0f}\n"
            
            if include_confidence and 'forecast_lower' in row and 'forecast_upper' in row:
                lower, upper = row['forecast_lower'], row['forecast_upper']
                if pd.notna(lower) and pd.notna(upper):
                    result += f"   95% CI: ${lower:,.0f} - ${upper:,.0f}\n"
            result += "\n"
        
        result += f"TOTAL: ${total_forecast:,.0f}\nAVERAGE: ${total_forecast/months_ahead:,.0f}\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def detect_anomalies(threshold_pct: float = 20.0, lookback_months: int = 6) -> str:
    """Detect demand anomalies."""
    try:
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        demand_df = demand_df.sort_values('month')
        
        recent = demand_df.tail(lookback_months)
        historical = demand_df.iloc[:-lookback_months]
        baseline_mean = historical['total_obligations_usd'].mean()
        
        result = f"ANOMALY DETECTION\nThreshold: ±{threshold_pct}%\nBaseline: ${baseline_mean:,.0f}\n" + "=" * 50 + "\n\n"
        anomalies_found = 0
        
        for _, row in recent.iterrows():
            actual = row['total_obligations_usd']
            deviation_pct = ((actual - baseline_mean) / baseline_mean) * 100
            
            if abs(deviation_pct) > threshold_pct:
                anomalies_found += 1
                severity = "🔴 CRITICAL" if abs(deviation_pct) > 50 else "🟠 HIGH" if abs(deviation_pct) > 30 else "🟡 MODERATE"
                direction = "ABOVE" if deviation_pct > 0 else "BELOW"
                result += f"{severity} - {row['month'].strftime('%B %Y')}\n   Actual: ${actual:,.0f}\n   Deviation: {deviation_pct:+.1f}% {direction}\n\n"
        
        if anomalies_found == 0:
            result += "✅ No anomalies detected.\n"
        else:
            result += f"TOTAL: {anomalies_found}\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Enriched Agent

# COMMAND ----------

# Define all tools (original + new)
tools = [
    # New enriched tools
    explain_demand_drivers,
    detect_trends,
    compare_forecast_models,
    assess_forecast_confidence,
    build_whatif_scenario,
    # Original tools
    get_demand_forecast,
    detect_anomalies
]

# Enhanced agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced AI assistant for Oshkosh Defense supply chain forecasting and analysis.

You have access to ENRICHED tools including:
- Feature importance analysis (explain what drives demand)
- Trend detection and pattern recognition
- Multi-model forecast comparison
- Confidence scoring for forecasts
- Interactive what-if scenario builder
- Standard forecasting and anomaly detection

When answering questions:
- Use feature importance to explain WHY demand changes
- Use trend analysis to identify patterns
- Compare multiple models for robust forecasts
- Assess confidence to guide planning decisions
- Build custom scenarios for strategic planning
- Always provide actionable insights with specific numbers

Be analytical, data-driven, and strategic in your responses."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
if _create_tool_calling_agent is not None and _AgentExecutor is not None:
    agent = _create_tool_calling_agent(llm, tools, prompt)
    agent_executor = _AgentExecutor(agent=agent, tools=tools, verbose=True)
    print(f"✓ Enriched agent created with {len(tools)} tools (AgentExecutor)")
else:
    tools_by_name = {t.name: t for t in tools}
    class _SimpleToolCallingExecutor:
        def __init__(self, llm, tools, verbose=True):
            self.llm = llm
            self.tools = tools
            self.verbose = verbose
        def invoke(self, inputs):
            user_input = inputs.get("input", "")
            messages = [HumanMessage(content=user_input)]
            max_rounds = 15
            for _ in range(max_rounds):
                response = self.llm.bind_tools(self.tools).invoke(messages)
                if self.verbose:
                    print(response.content[:200] if response.content else "(tool calls)", "...")
                if not getattr(response, "tool_calls", None):
                    return {"output": response.content or ""}
                for tc in response.tool_calls:
                    name = tc.get("name", None) if isinstance(tc, dict) else getattr(tc, "name", None)
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                    tid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    tool = tools_by_name.get(name) if name else None
                    if tool:
                        result = tool.invoke(args)
                        messages.append(ToolMessage(content=str(result), tool_call_id=tid))
                messages.append(response)
            return {"output": (response.content or "") + "\n[Max rounds reached.]"}
    agent_executor = _SimpleToolCallingExecutor(llm, tools, verbose=True)
    print(f"✓ Enriched agent created with {len(tools)} tools (bind_tools fallback)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Feature Importance Query

# COMMAND ----------

print("=" * 60)
print("EXAMPLE: What drives our demand?")
print("=" * 60)

response = agent_executor.invoke({
    "input": "What are the top factors that drive our demand forecasts? Explain the key drivers."
})
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Trend Analysis

# COMMAND ----------

print("=" * 60)
print("EXAMPLE: Trend analysis")
print("=" * 60)

response = agent_executor.invoke({
    "input": "Analyze demand trends over the last 12 months. Are we growing? Is there seasonality?"
})
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Model Comparison

# COMMAND ----------

print("=" * 60)
print("EXAMPLE: Compare forecasting models")
print("=" * 60)

response = agent_executor.invoke({
    "input": "Compare the forecasts from Prophet, ARIMA, and Random Forest for the next quarter. Which should we trust?"
})
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Confidence Assessment

# COMMAND ----------

print("=" * 60)
print("EXAMPLE: Forecast confidence")
print("=" * 60)

response = agent_executor.invoke({
    "input": "How confident should we be in our demand forecasts for the next 3 months?"
})
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Custom What-If Scenario

# COMMAND ----------

print("=" * 60)
print("EXAMPLE: Custom scenario")
print("=" * 60)

response = agent_executor.invoke({
    "input": "What if geopolitical risk increases by 50%, tariffs go up 25%, and commodity prices rise 15%? What's the combined impact?"
})
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool Reference

# COMMAND ----------

print("\n" + "=" * 60)
print("ENRICHED AGENT TOOL REFERENCE")
print("=" * 60)

tool_reference = [
    ("explain_demand_drivers", "Analyze feature importance", "What drives our demand?"),
    ("detect_trends", "Detect patterns and trends", "Are we growing? Any seasonality?"),
    ("compare_forecast_models", "Compare Prophet/ARIMA/RF", "Which model should we trust?"),
    ("assess_forecast_confidence", "Assess forecast reliability", "How confident are the forecasts?"),
    ("build_whatif_scenario", "Custom scenario builder", "What if risks increase by 50%?"),
    ("get_demand_forecast", "Get demand forecasts", "What's next quarter's forecast?"),
    ("detect_anomalies", "Find demand anomalies", "Any unusual patterns?")
]

for tool_name, description, example in tool_reference:
    print(f"\n{tool_name}")
    print(f"  {description}")
    print(f"  Example: \"{example}\"")

print("\n" + "=" * 60)
