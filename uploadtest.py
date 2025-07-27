import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Load and preprocess data
df = pd.read_excel("sample_power_demand_data.xlsx", engine='openpyxl')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df.sort_values(by=['State', 'Datetime'], inplace=True)

# Streamlit UI
st.title("🔮 Power Demand Forecasting")

# Sidebar: Model selection
model_type = st.sidebar.selectbox("Choose Forecasting Model", ["LSTM", "GRU"])

# Sidebar: Accuracy metrics placeholder
st.sidebar.subheader("📊 Accuracy Metrics")

# Main: State selection
state = st.selectbox("Select State", df['State'].unique())

# Filter and prepare data
state_df = df[df['State'] == state].sort_values('Datetime')
series = state_df['Power Demand (MW)'].values[:100]
train, test = series[:70], series[70:]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1))
test_scaled = scaler.transform(test.reshape(-1, 1))

# Create sequences
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

# Build and train model
model = Sequential()
if model_type == "LSTM":
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
else:
    model.add(GRU(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, verbose=0)

# Forecast
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Accuracy metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
r2 = r2_score(y_test_rescaled, predictions_rescaled)

# Display metrics in sidebar
st.sidebar.write(f"**RMSE**: {rmse:.2f}")
st.sidebar.write(f"**MAE**: {mae:.2f}")
st.sidebar.write(f"**R² Score**: {r2:.2f}")

# Plot results
st.subheader(f"📈 Forecast vs Actual using {model_type}")
plot_df = pd.DataFrame({
    'Datetime': state_df['Datetime'].values[80:100],
    'Actual': y_test_rescaled.flatten(),
    'Predicted': predictions_rescaled.flatten()
})

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=plot_df, x='Datetime', y='Actual', label='Actual', ax=ax)
sns.lineplot(data=plot_df, x='Datetime', y='Predicted', label='Predicted', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
