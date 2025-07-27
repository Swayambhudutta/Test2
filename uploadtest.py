import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

st.title("🔮 Power Demand Forecasting (Lightweight Model)")

uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df.sort_values(by=['State', 'Datetime'], inplace=True)

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df)

        state = "Maharashtra"
        state_df = df[df['State'] == state].sort_values('Datetime')
        series = state_df['Power Demand (MW)'].values[:100]
        train, test = series[:70], series[70:]

        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=30)

        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        r2 = r2_score(test, forecast)

        st.subheader("📊 Accuracy Metrics")
        st.write(f"**RMSE**: {rmse:.2f}")
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**R² Score**: {r2:.2f}")

        st.subheader("📈 Forecast vs Actual")
        plot_df = pd.DataFrame({
            'Datetime': state_df['Datetime'].values[70:100],
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
    st.info("Please upload an Excel file to begin analysis.")
