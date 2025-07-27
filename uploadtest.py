import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and preprocess data
df = pd.read_excel("sample_power_demand_data.xlsx", engine='openpyxl')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df.sort_values(by=['State', 'Datetime'], inplace=True)

# Streamlit UI
st.title("🔮 Power Demand Forecasting")

# Sidebar: Model selection
model_type = st.sidebar.selectbox("Choose Forecasting Model", ["SARIMAX", "RandomForest"])
st.sidebar.subheader("📊 Accuracy Metrics")

# Main: State selection
state = st.selectbox("Select State", df['State'].unique())

# Filter and prepare data
state_df = df[df['State'] == state].sort_values('Datetime')
series = state_df['Power Demand (MW)'].values[:100]
train, test = series[:70], series[70:]

# Forecasting
if model_type == "SARIMAX":
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
else:
    def create_features(data, window=5):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    window = 5
    X_train, y_train = create_features(scaled_series[:70], window)
    X_test, y_test = create_features(scaled_series[70-window:100], window)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    forecast_scaled = rf_model.predict(X_test)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Accuracy metrics
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
r2 = r2_score(test, forecast)

# Display metrics in sidebar
st.sidebar.write(f"**RMSE**: {rmse:.2f}")
st.sidebar.write(f"**MAE**: {mae:.2f}")
st.sidebar.write(f"**R² Score**: {r2:.2f}")

# Plot results
st.subheader(f"📈 Forecast vs Actual using {model_type}")
plot_df = pd.DataFrame({
    'Datetime': state_df['Datetime'].values[100 - len(test):100],
    'Actual': test,
    'Predicted': forecast
})

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=plot_df, x='Datetime', y='Actual', label='Actual', ax=ax)
sns.lineplot(data=plot_df, x='Datetime', y='Predicted', label='Predicted', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
