import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

st.title("🔮 Power Demand Forecasting")

# Sidebar: Model selection
model_type = st.sidebar.selectbox("Choose Forecasting Model", ["SARIMAX", "RandomForest", "LinearRegression", "SVR", "XGBoost", "LSTM", "GRU", "Hybrid"])
st.sidebar.subheader("📊 Accuracy Metrics")

# File uploader
uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df.sort_values(by=['State', 'Datetime'], inplace=True)

        state = st.selectbox("Select State", df['State'].unique())
        state_df = df[df['State'] == state].sort_values('Datetime')
        series = state_df['Power Demand (MW)'].values[:100]
        train, test = series[:70], series[70:]

        def create_features(data, window=5):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        if model_type == "SARIMAX":
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=30)
        elif model_type in ["RandomForest", "LinearRegression", "SVR", "XGBoost"]:
            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
            window = 5
            X_train, y_train = create_features(scaled_series[:70], window)
            X_test, y_test = create_features(scaled_series[70-window:100], window)

            if model_type == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "LinearRegression":
                model = LinearRegression()
            elif model_type == "SVR":
                model = SVR(kernel='rbf')
            elif model_type == "XGBoost":
                model = XGBRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            forecast_scaled = model.predict(X_test)
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        elif model_type in ["LSTM", "GRU", "Hybrid"]:
            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
            window = 5
            X_train, y_train = create_features(scaled_series[:70], window)
            X_test, y_test = create_features(scaled_series[70-window:100], window)

            X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
            y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
            X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

            class TimeSeriesModel(nn.Module):
                def __init__(self, model_type):
                    super().__init__()
                    if model_type == "LSTM":
                        self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                    elif model_type == "GRU":
                        self.rnn = nn.GRU(input_size=1, hidden_size=50, batch_first=True)
                    elif model_type == "Hybrid":
                        self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                        self.fc1 = nn.Linear(50, 25)
                        self.fc2 = nn.Linear(25, 1)
                    else:
                        raise ValueError("Invalid model type")

                    if model_type != "Hybrid":
                        self.fc = nn.Linear(50, 1)

                    self.model_type = model_type

                def forward(self, x):
                    out, _ = self.rnn(x)
                    out = out[:, -1, :]
                    if self.model_type == "Hybrid":
                        out = torch.relu(self.fc1(out))
                        out = self.fc2(out)
                    else:
                        out = self.fc(out)
                    return out

            model = TimeSeriesModel(model_type)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                output = model(X_train_torch)
                loss = criterion(output, y_train_torch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                forecast_scaled = model(X_test_torch).squeeze().numpy()

            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Accuracy metrics
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        r2_raw = r2_score(test, forecast)
        r2 = max(0.0, r2_raw)

        st.sidebar.write(f"**RMSE**: {rmse:.2f}")
        st.sidebar.write(f"**MAE**: {mae:.2f}")
        st.sidebar.write(f"**R² Score**: {r2:.2f}")

        # Insights section
        st.sidebar.subheader("💡 Insights")
        if r2_raw > 0.85 and rmse < 100 and mae < 100:
            st.sidebar.success("✅ The model performs well and is suitable for forecasting.")
            st.sidebar.markdown("""
            - High R² indicates strong correlation between predictions and actual values.
            - Low RMSE and MAE suggest minimal prediction error.
            - Model is reliable for short-term forecasting.
            """)
        elif r2_raw > 0.7:
            st.sidebar.warning("⚠️ The model shows moderate accuracy. Consider tuning parameters or using more data.")
            st.sidebar.markdown("""
            - R² is acceptable but not ideal for critical forecasting.
            - RMSE and MAE may still be high, indicating room for improvement.
            - Try increasing training data or adjusting model parameters.
            - Consider feature engineering or using ensemble methods.
            """)
        else:
            st.sidebar.error("❌ The model may not be reliable. Consider alternative models or preprocessing.")
            st.sidebar.markdown("""
            - R² is very low or negative, indicating poor predictive power.
            - High RMSE and MAE suggest significant prediction errors.
            - Model may be underfitting or missing key patterns.
            - Try using more advanced models or improving data quality.
            - Consider time series decomposition or external regressors.
            """)

        # Forecast vs Actual plot
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

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload an Excel file to proceed.")
