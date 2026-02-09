# Databricks notebook source
# MAGIC %md
# MAGIC # AI Agent Tools for Supply Chain Forecasting
# MAGIC
# MAGIC **Executive summary:** Exposes tools (forecast, anomalies, scenarios, DoD metrics, commodity prices) via a LangChain agent backed by Databricks Foundation Models. Management: use for self-serve analytics and what-if scenarios; deploy as Model Serving or scheduled jobs.
# MAGIC
# MAGIC **Depends on:** Gold tables (demand signals, DoD metrics, geo/trade risk, commodity, weather) and `gold.prophet_forecasts`. Run transformation and at least one forecasting notebook first.
# MAGIC
# MAGIC This notebook implements AI agent tools for:
# MAGIC - Demand forecasting queries
# MAGIC - Anomaly detection
# MAGIC - Scenario analysis (geopolitical, tariff, weather)
# MAGIC - DoD metrics comparison
# MAGIC - Commodity price monitoring
# MAGIC
# MAGIC **Framework**: LangChain with Databricks Foundation Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Install required packages. Use langchain>=0.2 so tool_calling_agent submodule exists (fixes ModuleNotFoundError).
%pip install --upgrade "typing_extensions>=4.1" "langchain>=0.2,<0.4" "langchain-core>=0.2" langgraph databricks-langchain mlflow pandas numpy

# COMMAND ----------

# Restart Python so upgraded typing_extensions is loaded (fixes ImportError: cannot import name 'Sentinel').
try:
    from typing_extensions import Sentinel
except ImportError:
    dbutils.library.restartPython()

# COMMAND ----------

# Do NOT use initialize_agent or AgentType from langchain.agents — they are deprecated/removed.
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
import mlflow
import json

# COMMAND ----------

# Configuration - Unity Catalog
CATALOG = "supply_chain"
DEMAND_SIGNALS_TABLE = f"{CATALOG}.gold.oshkosh_monthly_demand_signals"
DOD_METRICS_TABLE = f"{CATALOG}.gold.dod_metrics_inputs_monthly"
# Optional: geopolitical_risk_scores_monthly (from GDELT ingestion—removed; unified demand signals still have geo_risk_index)
GEO_RISK_TABLE = f"{CATALOG}.gold.geopolitical_risk_scores_monthly"
TRADE_RISK_TABLE = f"{CATALOG}.gold.trade_tariff_risk_monthly"
COMMODITY_TABLE = f"{CATALOG}.silver.commodity_prices_monthly"
WEATHER_TABLE = f"{CATALOG}.silver.weather_risk_monthly"
PROPHET_FORECAST_TABLE = f"{CATALOG}.gold.prophet_forecasts"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize LLM

# COMMAND ----------

# Initialize Databricks Foundation Model (Llama 3.3 70B - pay-per-token)
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.1,
    max_tokens=1000
)

# Enable MLflow tracing for agent interactions
mlflow.langchain.autolog()

print("LLM initialized: databricks-meta-llama-3-3-70b-instruct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 1: Demand Forecast Query

# COMMAND ----------

@tool
def get_demand_forecast(months_ahead: int = 3, include_confidence: bool = True) -> str:
    """
    Retrieve demand forecast for Oshkosh Defense contracts.
    
    Args:
        months_ahead: Number of months to forecast (1-12)
        include_confidence: Whether to include confidence intervals
    
    Returns:
        Forecast summary with predicted demand and confidence intervals
    """
    try:
        # Load forecast data
        forecast_df = spark.table(PROPHET_FORECAST_TABLE).toPandas()
        
        # Get future forecasts only
        forecast_df['month'] = pd.to_datetime(forecast_df['month'])
        current_date = datetime.now()
        
        future_forecasts = forecast_df[forecast_df['month'] > current_date].head(months_ahead)
        
        if future_forecasts.empty:
            return "No forecast data available. Please run the forecasting notebooks first."
        
        result = f"DEMAND FORECAST - Next {months_ahead} Months\n"
        result += "=" * 50 + "\n\n"
        
        total_forecast = 0
        for _, row in future_forecasts.iterrows():
            month_str = row['month'].strftime('%B %Y')
            forecast = row['forecast_demand_usd']
            total_forecast += forecast
            
            result += f"📅 {month_str}\n"
            result += f"   Forecast: ${forecast:,.0f}\n"
            
            if include_confidence and 'forecast_lower' in row and 'forecast_upper' in row:
                lower = row['forecast_lower']
                upper = row['forecast_upper']
                if pd.notna(lower) and pd.notna(upper):
                    result += f"   95% CI: ${lower:,.0f} - ${upper:,.0f}\n"
            result += "\n"
        
        result += f"TOTAL FORECAST: ${total_forecast:,.0f}\n"
        result += f"AVERAGE MONTHLY: ${total_forecast/months_ahead:,.0f}\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving forecast: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 2: Anomaly Detection

# COMMAND ----------

@tool
def detect_anomalies(threshold_pct: float = 20.0, lookback_months: int = 6) -> str:
    """
    Detect demand anomalies by comparing recent actuals to historical patterns.
    
    Args:
        threshold_pct: Percentage deviation to flag as anomaly (default 20%)
        lookback_months: Number of months to analyze (default 6)
    
    Returns:
        List of detected anomalies with severity levels
    """
    try:
        # Load demand signals
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        demand_df = demand_df.sort_values('month')
        
        # Get recent months
        recent = demand_df.tail(lookback_months)
        
        # Calculate historical baseline (excluding recent)
        historical = demand_df.iloc[:-lookback_months]
        baseline_mean = historical['total_obligations_usd'].mean()
        baseline_std = historical['total_obligations_usd'].std()
        
        result = f"ANOMALY DETECTION REPORT\n"
        result += f"Threshold: ±{threshold_pct}% from baseline\n"
        result += f"Baseline Mean: ${baseline_mean:,.0f}\n"
        result += "=" * 50 + "\n\n"
        
        anomalies_found = 0
        
        for _, row in recent.iterrows():
            month_str = row['month'].strftime('%B %Y')
            actual = row['total_obligations_usd']
            deviation_pct = ((actual - baseline_mean) / baseline_mean) * 100
            
            if abs(deviation_pct) > threshold_pct:
                anomalies_found += 1
                
                # Determine severity
                if abs(deviation_pct) > 50:
                    severity = "🔴 CRITICAL"
                elif abs(deviation_pct) > 30:
                    severity = "🟠 HIGH"
                else:
                    severity = "🟡 MODERATE"
                
                direction = "ABOVE" if deviation_pct > 0 else "BELOW"
                
                result += f"{severity} - {month_str}\n"
                result += f"   Actual: ${actual:,.0f}\n"
                result += f"   Deviation: {deviation_pct:+.1f}% {direction} baseline\n"
                result += f"   Risk Index: {row.get('combined_risk_index', 'N/A')}\n\n"
        
        if anomalies_found == 0:
            result += "✅ No anomalies detected within the specified threshold.\n"
        else:
            result += f"\nTOTAL ANOMALIES: {anomalies_found}\n"
        
        return result
        
    except Exception as e:
        return f"Error detecting anomalies: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 3: Geopolitical Risk Scenario

# COMMAND ----------

@tool
def scenario_geopolitical_risk(risk_level: str = "HIGH", region: str = "ALL") -> str:
    """
    Analyze impact of geopolitical risk scenarios on demand.
    
    Args:
        risk_level: Risk level to simulate (MODERATE, ELEVATED, HIGH, CRITICAL)
        region: Region to analyze (EUROPE, MIDEAST, INDO_PACIFIC, AMERICAS, ALL)
    
    Returns:
        Scenario analysis with demand impact projections
    """
    try:
        # Load demand data (required)
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        # Geopolitical risk table is optional (GDELT ingestion was removed; unified signals still have geo_risk_index)
        try:
            geo_risk_df = spark.table(GEO_RISK_TABLE).toPandas()
        except Exception:
            geo_risk_df = None

        # Define demand multipliers based on risk level
        multipliers = {
            "MODERATE": 1.0,
            "ELEVATED": 1.15,
            "HIGH": 1.35,
            "CRITICAL": 1.75
        }
        
        multiplier = multipliers.get(risk_level.upper(), 1.0)
        
        # Get baseline demand (last 12 months average)
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        recent_demand = demand_df.tail(12)['total_obligations_usd'].mean()
        
        # Calculate scenario impact
        scenario_demand = recent_demand * multiplier
        demand_increase = scenario_demand - recent_demand
        
        result = f"GEOPOLITICAL RISK SCENARIO ANALYSIS\n"
        result += "=" * 50 + "\n\n"
        result += f"Scenario: {risk_level.upper()} geopolitical risk\n"
        result += f"Region: {region.upper()}\n\n"
        
        result += f"BASELINE (Current):\n"
        result += f"   Average Monthly Demand: ${recent_demand:,.0f}\n\n"
        
        result += f"SCENARIO PROJECTION:\n"
        result += f"   Demand Multiplier: {multiplier:.2f}x\n"
        result += f"   Projected Monthly Demand: ${scenario_demand:,.0f}\n"
        result += f"   Monthly Increase: ${demand_increase:,.0f} ({(multiplier-1)*100:.0f}%)\n"
        result += f"   Annual Impact: ${demand_increase * 12:,.0f}\n\n"
        
        result += f"RECOMMENDED ACTIONS:\n"
        if risk_level.upper() == "CRITICAL":
            result += "   🔴 Activate surge capacity protocols\n"
            result += "   🔴 Increase safety stock by 75%\n"
            result += "   🔴 Expedite critical supplier orders\n"
            result += "   🔴 Review alternative supplier options\n"
        elif risk_level.upper() == "HIGH":
            result += "   🟠 Increase safety stock by 35%\n"
            result += "   🟠 Accelerate procurement timelines\n"
            result += "   🟠 Monitor supplier capacity closely\n"
        elif risk_level.upper() == "ELEVATED":
            result += "   🟡 Increase safety stock by 15%\n"
            result += "   🟡 Review supplier contingency plans\n"
        else:
            result += "   🟢 Continue normal operations\n"
            result += "   🟢 Maintain standard safety stock levels\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing geopolitical scenario: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 4: Tariff Risk Scenario

# COMMAND ----------

@tool
def scenario_tariff_increase(tariff_increase_pct: float = 25.0, product_category: str = "ALL") -> str:
    """
    Analyze impact of tariff increases on supply chain costs.
    
    Args:
        tariff_increase_pct: Percentage increase in tariffs (default 25%)
        product_category: Product category (VEHICLES, ELECTRONICS, STEEL, ALUMINUM, ALL)
    
    Returns:
        Cost impact analysis with mitigation recommendations
    """
    try:
        # Load data
        demand_df = spark.table(DEMAND_SIGNALS_TABLE).toPandas()
        trade_risk_df = spark.table(TRADE_RISK_TABLE).toPandas()
        
        # Get baseline spend
        demand_df['month'] = pd.to_datetime(demand_df['month'])
        recent_demand = demand_df.tail(12)['total_obligations_usd'].mean()
        
        # Estimate import content (assumption: 30% of spend is imported materials)
        import_content_pct = 0.30
        imported_value = recent_demand * import_content_pct
        
        # Calculate tariff impact
        tariff_cost_increase = imported_value * (tariff_increase_pct / 100)
        total_cost_increase_pct = (tariff_cost_increase / recent_demand) * 100
        
        result = f"TARIFF INCREASE SCENARIO ANALYSIS\n"
        result += "=" * 50 + "\n\n"
        result += f"Scenario: {tariff_increase_pct}% tariff increase\n"
        result += f"Product Category: {product_category.upper()}\n\n"
        
        result += f"BASELINE:\n"
        result += f"   Monthly Spend: ${recent_demand:,.0f}\n"
        result += f"   Estimated Import Content: {import_content_pct*100:.0f}%\n"
        result += f"   Imported Value: ${imported_value:,.0f}\n\n"
        
        result += f"TARIFF IMPACT:\n"
        result += f"   Additional Tariff Cost: ${tariff_cost_increase:,.0f}/month\n"
        result += f"   Total Cost Increase: {total_cost_increase_pct:.1f}%\n"
        result += f"   Annual Impact: ${tariff_cost_increase * 12:,.0f}\n\n"
        
        result += f"MITIGATION STRATEGIES:\n"
        if tariff_increase_pct >= 50:
            result += "   🔴 Urgent: Evaluate domestic sourcing alternatives\n"
            result += "   🔴 Negotiate long-term contracts before increase\n"
            result += "   🔴 Consider tariff exclusion applications\n"
        elif tariff_increase_pct >= 25:
            result += "   🟠 Accelerate supplier diversification\n"
            result += "   🟠 Review make-vs-buy decisions\n"
            result += "   🟠 Explore bonded warehouse options\n"
        else:
            result += "   🟡 Monitor tariff developments\n"
            result += "   🟡 Update cost models\n"
            result += "   🟡 Review pricing strategies\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing tariff scenario: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 5: Weather Disruption Scenario

# COMMAND ----------

@tool
def scenario_weather_disruption(disruption_type: str = "SEVERE_WINTER", affected_region: str = "MIDWEST") -> str:
    """
    Analyze impact of weather disruptions on supply chain.
    
    Args:
        disruption_type: Type of weather event (SEVERE_WINTER, HURRICANE, FLOODING, EXTREME_HEAT)
        affected_region: Region affected (MIDWEST, SOUTHEAST, GULF_COAST, WEST_COAST)
    
    Returns:
        Supply chain disruption analysis with contingency recommendations
    """
    try:
        # Load weather risk data
        weather_df = spark.table(WEATHER_TABLE).toPandas()
        
        # Define disruption impacts
        disruption_impacts = {
            "SEVERE_WINTER": {
                "transport_delay_days": 7,
                "production_impact_pct": 15,
                "affected_suppliers": ["POWERTRAIN", "SUSPENSION", "MATERIALS"]
            },
            "HURRICANE": {
                "transport_delay_days": 14,
                "production_impact_pct": 25,
                "affected_suppliers": ["ELECTRONICS", "TIRES", "ARMOR"]
            },
            "FLOODING": {
                "transport_delay_days": 10,
                "production_impact_pct": 20,
                "affected_suppliers": ["MATERIALS", "HYDRAULICS", "ELECTRICAL"]
            },
            "EXTREME_HEAT": {
                "transport_delay_days": 3,
                "production_impact_pct": 10,
                "affected_suppliers": ["RUBBER", "ELECTRONICS"]
            }
        }
        
        impact = disruption_impacts.get(disruption_type.upper(), disruption_impacts["SEVERE_WINTER"])
        
        result = f"WEATHER DISRUPTION SCENARIO ANALYSIS\n"
        result += "=" * 50 + "\n\n"
        result += f"Scenario: {disruption_type.replace('_', ' ')}\n"
        result += f"Affected Region: {affected_region.replace('_', ' ')}\n\n"
        
        result += f"ESTIMATED IMPACTS:\n"
        result += f"   Transportation Delays: {impact['transport_delay_days']} days\n"
        result += f"   Production Impact: {impact['production_impact_pct']}% reduction\n"
        result += f"   Affected Subsystems: {', '.join(impact['affected_suppliers'])}\n\n"
        
        result += f"SUPPLY CHAIN VULNERABILITIES:\n"
        for subsystem in impact['affected_suppliers']:
            result += f"   ⚠️ {subsystem}: Potential delays and shortages\n"
        
        result += f"\nCONTINGENCY ACTIONS:\n"
        if impact['transport_delay_days'] >= 10:
            result += "   🔴 Activate emergency logistics protocols\n"
            result += "   🔴 Pre-position critical inventory\n"
            result += "   🔴 Engage backup transportation providers\n"
        else:
            result += "   🟡 Monitor weather forecasts closely\n"
            result += "   🟡 Communicate with affected suppliers\n"
            result += "   🟡 Review safety stock levels\n"
        
        result += f"\nRECOVERY TIMELINE:\n"
        result += f"   Expected normalization: {impact['transport_delay_days'] + 7} days\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing weather scenario: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 6: DoD Metrics Comparison

# COMMAND ----------

@tool
def compare_dod_metrics(metric_type: str = "ALL") -> str:
    """
    Compare current performance against DoD supply chain metrics objectives.
    
    Args:
        metric_type: Metric to analyze (RO, AAO, DAYS_OF_SUPPLY, NMCS_RISK, ALL)
    
    Returns:
        DoD metrics comparison with performance assessment
    """
    try:
        # Load DoD metrics
        dod_df = spark.table(DOD_METRICS_TABLE).toPandas()
        dod_df['month'] = pd.to_datetime(dod_df['month'])
        
        # Get latest metrics
        latest = dod_df.sort_values('month').iloc[-1]
        
        result = f"DoD SUPPLY CHAIN METRICS ASSESSMENT\n"
        result += "=" * 50 + "\n"
        result += f"As of: {latest['month'].strftime('%B %Y')}\n\n"
        
        # Requirements Objective
        if metric_type.upper() in ["RO", "ALL"]:
            ro = latest['requirements_objective_proxy']
            risk_adj_ro = latest['risk_adjusted_ro']
            
            result += f"📊 REQUIREMENTS OBJECTIVE (RO)\n"
            result += f"   Current RO: ${ro:,.0f}\n"
            result += f"   Risk-Adjusted RO: ${risk_adj_ro:,.0f}\n"
            result += f"   Status: {'✅ Adequate' if ro > 0 else '⚠️ Review needed'}\n\n"
        
        # Approved Acquisition Objective
        if metric_type.upper() in ["AAO", "ALL"]:
            aao = latest['approved_acquisition_objective_proxy']
            
            result += f"📊 APPROVED ACQUISITION OBJECTIVE (AAO)\n"
            result += f"   Current AAO: ${aao:,.0f}\n"
            result += f"   (Includes 2-year forecast demand)\n\n"
        
        # Days of Supply
        if metric_type.upper() in ["DAYS_OF_SUPPLY", "ALL"]:
            dos = latest['days_of_supply_proxy']
            
            status = "✅ Healthy" if dos >= 60 else "🟡 Monitor" if dos >= 30 else "🔴 Critical"
            
            result += f"📊 DAYS OF SUPPLY\n"
            result += f"   Current: {dos:.0f} days\n"
            result += f"   Target: 60+ days\n"
            result += f"   Status: {status}\n\n"
        
        # NMCS Risk
        if metric_type.upper() in ["NMCS_RISK", "ALL"]:
            nmcs = latest['nmcs_risk_indicator']
            
            result += f"📊 NOT MISSION CAPABLE - SUPPLY (NMCS) RISK\n"
            result += f"   Current Risk Level: {nmcs}\n"
            if nmcs == "HIGH_RISK":
                result += f"   ⚠️ Action Required: Increase safety stock immediately\n"
            elif nmcs == "ELEVATED_RISK":
                result += f"   🟡 Monitor closely and review inventory levels\n"
            else:
                result += f"   ✅ Within acceptable parameters\n"
            result += "\n"
        
        # Demand Volatility
        if metric_type.upper() == "ALL":
            volatility = latest['demand_volatility_category']
            cv = latest['coefficient_of_variation']
            
            result += f"📊 DEMAND VOLATILITY\n"
            result += f"   Category: {volatility}\n"
            result += f"   Coefficient of Variation: {cv:.2f}\n"
            result += f"   Recommended Forecast Method: {latest['forecast_method_recommendation']}\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving DoD metrics: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 7: Commodity Price Monitor

# COMMAND ----------

@tool
def get_commodity_prices(category: str = "ALL") -> str:
    """
    Get current prices for defense-critical commodities.
    
    Args:
        category: Commodity category (ENERGY, PRECIOUS_METALS, INDUSTRIAL_METALS, BATTERY_MATERIALS, ALL)
    
    Returns:
        Commodity price summary with supply chain impact analysis
    """
    try:
        # Load commodity prices
        commodity_df = spark.table(COMMODITY_TABLE).toPandas()
        commodity_df['month'] = pd.to_datetime(commodity_df['month'])
        
        # Get latest prices
        latest_month = commodity_df['month'].max()
        latest = commodity_df[commodity_df['month'] == latest_month]
        
        if category.upper() != "ALL":
            latest = latest[latest['category'].str.upper() == category.upper()]
        
        result = f"DEFENSE CRITICAL MATERIALS PRICE MONITOR\n"
        result += "=" * 50 + "\n"
        result += f"As of: {latest_month.strftime('%B %Y')}\n\n"
        
        # Group by category
        for cat in latest['category'].unique():
            cat_data = latest[latest['category'] == cat]
            
            result += f"📊 {cat.upper()}\n"
            for _, row in cat_data.iterrows():
                trend = "🔺" if row['pct_change_1mo'] > 0 else "🔻"
                result += f"   • {row['commodity_name']}: ${row['close_price']:,.2f}\n"
                result += f"     {trend} {row['pct_change_1mo']:+.1f}% (1mo) | {row['pct_change_3mo']:+.1f}% (3mo)\n"
                result += f"     Use: {row['defense_use']}\n"
            result += "\n"
        
        # Cost pressure summary
        avg_pressure = latest['cost_pressure_score'].mean()
        
        result += f"COST PRESSURE ASSESSMENT:\n"
        if avg_pressure > 10:
            result += f"   🔴 HIGH PRESSURE: Average {avg_pressure:.1f} - Consider accelerating procurement\n"
        elif avg_pressure > 0:
            result += f"   🟡 MODERATE PRESSURE: Average {avg_pressure:.1f} - Monitor closely\n"
        else:
            result += f"   🟢 FAVORABLE: Average {avg_pressure:.1f} - Potential cost savings opportunity\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving commodity prices: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Agent

# COMMAND ----------

# Define all tools
tools = [
    get_demand_forecast,
    detect_anomalies,
    scenario_geopolitical_risk,
    scenario_tariff_increase,
    scenario_weather_disruption,
    compare_dod_metrics,
    get_commodity_prices
]

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for Oshkosh Defense supply chain forecasting and analysis.
    
You have access to tools that can:
1. Retrieve demand forecasts
2. Detect anomalies in demand patterns
3. Analyze geopolitical risk scenarios
4. Analyze tariff/trade risk scenarios
5. Analyze weather disruption scenarios
6. Compare against DoD supply chain metrics
7. Monitor commodity prices

When answering questions:
- Use the appropriate tool to get data
- Provide clear, actionable insights
- Reference DoD metrics and terminology where applicable
- Highlight risks and recommended actions

Always be specific with numbers and dates when available."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent (use LangChain AgentExecutor if available; else simple bind_tools loop)
if _create_tool_calling_agent is not None and _AgentExecutor is not None:
    agent = _create_tool_calling_agent(llm, tools, prompt)
    agent_executor = _AgentExecutor(agent=agent, tools=tools, verbose=True)
    print("Agent created with 7 tools (LangChain AgentExecutor)")
else:
    # Fallback when create_tool_calling_agent / create_react_agent / AgentExecutor not in langchain.agents
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
    print("Agent created with 7 tools (bind_tools fallback)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Agent Interactions

# COMMAND ----------

# Example 1: Forecast query
print("=" * 60)
print("EXAMPLE 1: Forecast Query")
print("=" * 60)

response = agent_executor.invoke({
    "input": "What is the demand forecast for the next quarter?"
})
print(response["output"])

# COMMAND ----------

# Example 2: Anomaly detection
print("=" * 60)
print("EXAMPLE 2: Anomaly Detection")
print("=" * 60)

response = agent_executor.invoke({
    "input": "Are there any demand anomalies in the last 6 months?"
})
print(response["output"])

# COMMAND ----------

# Example 3: Geopolitical scenario
print("=" * 60)
print("EXAMPLE 3: Geopolitical Risk Scenario")
print("=" * 60)

response = agent_executor.invoke({
    "input": "What would happen to demand if geopolitical risk becomes CRITICAL in Europe?"
})
print(response["output"])

# COMMAND ----------

# Example 4: DoD metrics
print("=" * 60)
print("EXAMPLE 4: DoD Metrics Comparison")
print("=" * 60)

response = agent_executor.invoke({
    "input": "How do our current metrics compare to DoD objectives?"
})
print(response["output"])

# COMMAND ----------

# Example 5: Commodity prices
print("=" * 60)
print("EXAMPLE 5: Commodity Price Check")
print("=" * 60)

response = agent_executor.invoke({
    "input": "What are the current prices for industrial metals and how might they affect our costs?"
})
print(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Agent Configuration

# COMMAND ----------

# Save agent configuration for deployment
agent_config = {
    "model": "databricks-meta-llama-3-3-70b-instruct",
    "temperature": 0.1,
    "max_tokens": 1000,
    "tools": [t.name for t in tools],
    "data_sources": {
        "demand_signals": DEMAND_SIGNALS_TABLE,
        "dod_metrics": DOD_METRICS_TABLE,
        "geo_risk": GEO_RISK_TABLE,
        "trade_risk": TRADE_RISK_TABLE,
        "commodity": COMMODITY_TABLE,
        "weather": WEATHER_TABLE,
        "forecasts": PROPHET_FORECAST_TABLE
    },
    "created_at": datetime.now().isoformat()
}

# Save agent config (local only; DBFS/workspace copy skipped when local filesystem access is forbidden)
import json
with open("/tmp/agent_config.json", "w") as f:
    json.dump(agent_config, f, indent=2)
print("Agent configuration saved to /tmp/agent_config.json")
# Optional: copy to DBFS only when allowed (many clusters forbid WorkspaceLocalFileSystem/DBFS root)
# try:
#     dbutils.fs.cp("file:/tmp/agent_config.json", "dbfs:/models/agent_config.json")
#     print("Also copied to dbfs:/models/agent_config.json")
# except Exception as e:
#     print(f"DBFS copy skipped (restricted): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Tool Reference
# MAGIC
# MAGIC | Tool | Description | Example Query |
# MAGIC |------|-------------|---------------|
# MAGIC | `get_demand_forecast` | Retrieve demand forecasts | "What's the forecast for next quarter?" |
# MAGIC | `detect_anomalies` | Find demand anomalies | "Are there any unusual demand patterns?" |
# MAGIC | `scenario_geopolitical_risk` | Analyze geo risk impact | "What if geo risk becomes CRITICAL?" |
# MAGIC | `scenario_tariff_increase` | Analyze tariff impact | "What's the impact of 25% tariff increase?" |
# MAGIC | `scenario_weather_disruption` | Analyze weather impact | "How would a severe winter affect supply?" |
# MAGIC | `compare_dod_metrics` | Check DoD metrics | "How do we compare to DoD objectives?" |
# MAGIC | `get_commodity_prices` | Monitor commodity costs | "What are current steel prices?" |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Deploy agent as a Databricks Model Serving endpoint
# MAGIC 2. Create Databricks Jobs for scheduled data refreshes
# MAGIC 3. Build dashboard for visualization