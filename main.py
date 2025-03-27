from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# API URL for fetching payment data
API_URL = "http://192.168.1.2:9999/api/payment_data"

# Function to fetch and filter data by company
def get_company_data(company_name):
    response = requests.get(API_URL)
    if response.status_code != 200:
        return None
    
    data = response.json()
    df = pd.DataFrame(data)
    print(df)
    # Convert to datetime
    df["creationDateTime"] = pd.to_datetime(df["creationDateTime"].apply(lambda x: "-".join(map(str, x[:3]))))
    
    # Filter by company
    company_data = df[(df["creditorName"] == company_name) | (df["debtorName"] == company_name)]
    
    return company_data if not company_data.empty else None

# Function to train LSTM model
def train_lstm(company_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    company_data["amount"] = scaler.fit_transform(company_data["amount"].values.reshape(-1, 1))

    # Prepare time series data
    X, y = [], []
    for i in range(len(company_data) - 1):
        X.append(company_data["amount"].iloc[i])
        y.append(company_data["amount"].iloc[i + 1])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 1, 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)
    
    return model, scaler, X[-1]  # Return last known value for predictions

# Function to predict future cash flow
def predict_cashflow(model, scaler, last_known_value, timeframe="daily", steps=7):
    predictions = []
    interval = {"hourly": "hours", "weekly": "weeks"}.get(timeframe, "days")
    future_dates = [datetime.datetime.now() + datetime.timedelta(**{interval: i}) for i in range(1, steps+1)]
    
    for date in future_dates:
        predicted_value = model.predict(last_known_value.reshape(1, 1, 1))[0][0]
        last_known_value = np.array([[predicted_value]])  # Update for next prediction
        predictions.append({"datetime": str(date), "predicted_cashflow": round(scaler.inverse_transform([[predicted_value]])[0][0], 2)})
    
    return predictions

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    company_name = data.get("companyName")
    timeframe = data.get("timeframe", "daily")  # hourly, daily, weekly

    if not company_name:
        return jsonify({"error": "Company name is required"}), 400

    company_data = get_company_data(company_name)
    if company_data is None:
        return jsonify({"error": f"No data found for {company_name}"}), 404

    model, scaler, last_known_value = train_lstm(company_data)
    predictions = predict_cashflow(model, scaler, last_known_value, timeframe, steps=24 if timeframe == "hourly" else 7 if timeframe == "daily" else 4)

    return jsonify({"company": company_name, "timeframe": timeframe, "predictions": predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)