import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image
import os

st.title("🔮 Power Demand Forecasting with LSTM")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df.sort_values(by=['State', 'Datetime'], inplace=True)

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df)

        # Prepare results container
        results = {}
        image_paths = []

        for state in df['State'].unique():
            state_df = df[df['State'] == state].sort_values('Datetime')
            demand_series = state_df['Power Demand (MW)'].values

            if len(demand_series) < 100:
                st.warning(f"Not enough data points for {state}. Skipping.")
                continue

            train_data = demand_series[:70]
            test_data = demand_series[70:100]

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
            test_scaled = scaler.transform(test_data.reshape(-1, 1))

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i+seq_length])
                    y.append(data[i+seq_length])
                return np.array(X), np.array(y)

            seq_length = 10
            X_train, y_train = create_sequences(train_scaled, seq_length)
            X_test, y_test = create_sequences(test_scaled, seq_length)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, verbose=0)

            predictions = model.predict(X_test)
            predictions_rescaled = scaler.inverse_transform(predictions)
            y_test_rescaled = scaler.inverse_transform(y_test)

            rmse = mean_squared_error(y_test_rescaled, predictions_rescaled, squared=False)
            mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
            r2 = r2_score(y_test_rescaled, predictions_rescaled)

            results[state] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

            # Plot and save
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test_rescaled, label='Actual')
            ax.plot(predictions_rescaled, label='Predicted')
            ax.set_title(f'LSTM Prediction vs Actual for {state}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Power Demand (MW)')
            ax.legend()
            image_path = f'lstm_comparison_{state}.png'
            fig.savefig(image_path)
            image_paths.append((state, image_path))
            plt.close()

        # Show metrics
        st.subheader("📊 Accuracy Metrics")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df)

        # Show plots
        st.subheader("📈 Prediction vs Actual")
        for state, path in image_paths:
            st.markdown(f"### {state}")
            st.image(Image.open(path), caption=f"LSTM Prediction vs Actual for {state}", use_column_width=True)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload an Excel file to begin analysis.")
