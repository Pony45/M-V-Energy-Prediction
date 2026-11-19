import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# 1. SETUP PAGE
st.set_page_config(page_title="AI Energy M&V Dashboard", layout="wide")
st.title("⚡ AI Energy Measurement & Verification (M&V)")
st.markdown("**Project:** Predicting Energy Saving using IPMVP Option C based on AI Techniques")

# 2. LOAD & PROCESS DATA
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Date'].dt.hour
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
    df['Occupancy'] = df['Occupancy'].map({'Ada Orang': 1, 'Tiada Orang': 0})
    return df

df = load_data()

# 3. SIMULATE M&V
df.loc[df['Date'].dt.month == 10, 'Total_kWh_Used'] *= 0.75 # Apply ECM
baseline_data = df[df['Date'].dt.month < 10]
ecm_data = df[df['Date'].dt.month == 10]
features = ['Temp_C', 'Humidity_Pct', 'Occupancy', 'Hour', 'Is_Weekend', 'Day_of_Week']

# PECAHKAN DATA BASELINE UNTUK METRIK
X_base = baseline_data[features]
y_base = baseline_data['Total_kWh_Used']
X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

# TRAIN MODEL UNTUK METRIK
model_xgb_metric = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_rf_metric = RandomForestRegressor(n_estimators=100, random_state=42)

model_xgb_metric.fit(X_train, y_train)
model_rf_metric.fit(X_train, y_train)

# KIRA METRIK
pred_xgb_metric = model_xgb_metric.predict(X_test)
pred_rf_metric = model_rf_metric.predict(X_test)

r2_xgb = r2_score(y_test, pred_xgb_metric)
mae_xgb = mean_absolute_error(y_test, pred_xgb_metric)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb_metric))

r2_rf = r2_score(y_test, pred_rf_metric)
mae_rf = mean_absolute_error(y_test, pred_rf_metric)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf_metric))

# TRAIN MODEL PENUH (Untuk kira Savings M&V macam biasa)
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

model_xgb.fit(X_base, y_base)
model_rf.fit(X_base, y_base)

# PREDICT BASELINE
ecm_data = ecm_data.copy()
ecm_data['Predicted_Baseline_XGB'] = model_xgb.predict(ecm_data[features])
ecm_data['Predicted_Baseline_RF'] = model_rf.predict(ecm_data[features])

# 4. CALCULATE KPIs (XGBoost)
total_actual = ecm_data['Total_kWh_Used'].sum()
total_baseline_xgb = ecm_data['Predicted_Baseline_XGB'].sum()
savings_xgb = total_baseline_xgb - total_actual
savings_pct_xgb = (savings_xgb / total_baseline_xgb) * 100

# CALCULATE KPIs (Random Forest)
total_baseline_rf = ecm_data['Predicted_Baseline_RF'].sum()
savings_rf = total_baseline_rf - total_actual
savings_pct_rf = (savings_rf / total_baseline_rf) * 100

# ==========================================
# BAHAGIAN 1: XGBOOST RESULTS
# ==========================================
st.header("🔴 1. XGBoost Model Results")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Baseline (XGB)", f"{total_baseline_xgb:.2f} kWh")
col2.metric("Actual Usage (ECM)", f"{total_actual:.2f} kWh")
col3.metric("Total Energy Savings (XGB)", f"{savings_xgb:.2f} kWh", f"{savings_pct_xgb:.2f}%")
# METRIK XGBOOST
st.markdown("**Model Performance Metrics (XGBoost):**")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.info(f"**R² Score:** {r2_xgb:.4f}")
m_col2.info(f"**MAE:** {mae_xgb:.4f} kWh")
m_col3.info(f"**RMSE:** {rmse_xgb:.4f} kWh")

plot_data_xgb = ecm_data.head(168)
fig_xgb = px.line(plot_data_xgb, x='Date', y=['Total_kWh_Used', 'Predicted_Baseline_XGB'],
              labels={'value': 'Energy (kWh)', 'variable': 'Legend'},
              title="Baseline vs Actual Usage (XGBoost - First 7 Days of Oct)",
              color_discrete_sequence=['#00CC96', '#EF553B'])
st.plotly_chart(fig_xgb, use_container_width=True)

st.markdown("---")

# ==========================================
# BAHAGIAN 2: RANDOM FOREST RESULTS
# ==========================================
st.header("🟣 2. Random Forest Model Results")
col1_rf, col2_rf, col3_rf = st.columns(3)
col1_rf.metric("Predicted Baseline (RF)", f"{total_baseline_rf:.2f} kWh")
col2_rf.metric("Actual Usage (ECM)", f"{total_actual:.2f} kWh")
col3_rf.metric("Total Energy Savings (RF)", f"{savings_rf:.2f} kWh", f"{savings_pct_rf:.2f}%")
# METRIK RANDOM FOREST
st.markdown("**Model Performance Metrics (Random Forest):**")
m_col1_rf, m_col2_rf, m_col3_rf = st.columns(3)
m_col1_rf.info(f"**R² Score:** {r2_rf:.4f}")
m_col2_rf.info(f"**MAE:** {mae_rf:.4f} kWh")
m_col3_rf.info(f"**RMSE:** {rmse_rf:.4f} kWh")

plot_data_rf = ecm_data.head(168)
fig_rf = px.line(plot_data_rf, x='Date', y=['Total_kWh_Used', 'Predicted_Baseline_RF'],
              labels={'value': 'Energy (kWh)', 'variable': 'Legend'},
              title="Baseline vs Actual Usage (Random Forest - First 7 Days of Oct)",
              color_discrete_sequence=['#00CC96', '#AB63FA']) # Warna ungu untuk RF
st.plotly_chart(fig_rf, use_container_width=True)

# SIDEBAR
st.sidebar.header("About")
st.sidebar.info("This dashboard visualizes the M&V process using IPMVP Option C. It compares the savings calculated by two AI models: XGBoost and Random Forest.")
