import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor

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
df.loc[df['Date'].dt.month == 10, 'Total_kWh_Used'] *= 0.75
baseline_data = df[df['Date'].dt.month < 10]
ecm_data = df[df['Date'].dt.month == 10]
features = ['Temp_C', 'Humidity_Pct', 'Occupancy', 'Hour', 'Is_Weekend', 'Day_of_Week']

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(baseline_data[features], baseline_data['Total_kWh_Used'])

# Predict baseline for October
ecm_data = ecm_data.copy()
ecm_data['Predicted_Baseline'] = model.predict(ecm_data[features])

# 4. CALCULATE KPIs
total_actual = ecm_data['Total_kWh_Used'].sum()
total_baseline = ecm_data['Predicted_Baseline'].sum()
total_savings = total_baseline - total_actual
savings_pct = (total_savings / total_baseline) * 100

# 5. DISPLAY METRICS
st.markdown("### 📊 Summary (October ECM Period)")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Baseline", f"{total_baseline:.2f} kWh")
col2.metric("Actual Usage (ECM)", f"{total_actual:.2f} kWh")
col3.metric("Total Energy Savings", f"{total_savings:.2f} kWh", f"{savings_pct:.2f}%")

st.markdown("---")

# 6. DISPLAY GRAPH
st.markdown("### 📈 Baseline vs Actual Usage (First 7 Days of October)")
plot_data = ecm_data.head(168) # 1 week
fig = px.line(plot_data, x='Date', y=['Total_kWh_Used', 'Predicted_Baseline'],
              labels={'value': 'Energy (kWh)', 'variable': 'Legend'},
              color_discrete_sequence=['#00CC96', '#EF553B']) # Hijau & Merah
fig.update_layout(legend_title_text='')
st.plotly_chart(fig, use_container_width=True)

# 7. SIDEBAR INFO
st.sidebar.header("About")
st.sidebar.info("This dashboard visualizes the Measurement & Verification (M&V) process using IPMVP Option C. Energy savings are calculated by comparing the AI-predicted baseline against the actual post-retrofit consumption.")
