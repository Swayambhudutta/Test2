import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

def create_dl_features(data, window=5):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Deep Learning Models
elif model_type in ["LSTM", "GRU", "Hybrid"]:
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    window = 5
    X_train, y_train = create_dl_features(scaled_series[:70], window)
    X_test, y_test = create_dl_features(scaled_series[70-window:100], window)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, activation='relu', input_shape=(window, 1)))
    elif model_type == "GRU":
        model.add(GRU(50, activation='relu', input_shape=(window, 1)))
    elif model_type == "Hybrid":
        model.add(LSTM(50, activation='relu', input_shape=(window, 1)))
        model.add(Dense(25, activation='relu'))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)

    forecast_scaled = model.predict(X_test).flatten()
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
