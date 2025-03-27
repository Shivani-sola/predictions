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


# Fetch and filter data by company
def get_company_data(company_name):
    response = requests.get(API_URL)
    if response.status_code != 200:
        return None

    data = response.json()
    df = pd.DataFrame(data)

    # Convert date
    df["creationDateTime"] = pd.to_datetime(df["creationDateTime"].apply(lambda x: "-".join(map(str, x[:3]))))

    # Filter for the given company
    company_data = df[(df["creditorName"] == company_name) | (df["debtorName"] == company_name)]

    if company_data.empty:
        return None

    # Separate inflow (credits) and outflow (debits)
    inflow_data = company_data[company_data["creditorName"] == company_name][["creationDateTime", "amount"]]
    outflow_data = company_data[company_data["debtorName"] == company_name][["creationDateTime", "amount"]]

    return inflow_data, outflow_data


# Train LSTM model
def train_lstm(data):
    if data.empty:
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    data["amount"] = scaler.fit_transform(data["amount"].values.reshape(-1, 1))

    # Prepare time series data
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data["amount"].iloc[i])
        y.append(data["amount"].iloc[i + 1])

    if len(X) == 0:
        return None, None, None

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


# Predict future values
def predict_cashflow(model, scaler, last_known_value, timeframe="daily", steps=7):
    if model is None:
        return [{"datetime": str(datetime.datetime.now()), "predicted_value": 0}]

    predictions = []
    interval = "days" if timeframe == "daily" else "weeks"
    steps = 7 if timeframe == "weekly" else 1

    future_dates = [datetime.datetime.now() + datetime.timedelta(**{interval: i}) for i in range(1, steps + 1)]

    for date in future_dates:
        predicted_value = model.predict(last_known_value.reshape(1, 1, 1))[0][0]
        last_known_value = np.array([[predicted_value]])

        actual_value = round(scaler.inverse_transform([[predicted_value]])[0][0], 2)
        predictions.append({"datetime": str(date), "predicted_value": actual_value})

    return predictions


# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    company_name = data.get("companyName")
    timeframe = data.get("timeframe", "daily")

    if not company_name:
        return jsonify({"error": "Company name is required"}), 400

    inflow_data, outflow_data = get_company_data(company_name)
    if inflow_data is None or outflow_data is None:
        return jsonify({"error": f"No data found for {company_name}"}), 404

    # Train models
    model_inflow, scaler_inflow, last_inflow = train_lstm(inflow_data)
    model_outflow, scaler_outflow, last_outflow = train_lstm(outflow_data)

    inflow_predictions = predict_cashflow(model_inflow, scaler_inflow, last_inflow, timeframe)
    outflow_predictions = predict_cashflow(model_outflow, scaler_outflow, last_outflow, timeframe)

    # Compute net cashflow
    net_cashflow = []
    for i in range(len(inflow_predictions)):
        inflow = inflow_predictions[i]["predicted_value"]
        outflow = outflow_predictions[i]["predicted_value"]
        net_cashflow.append({
            "datetime": inflow_predictions[i]["datetime"],
            "cash_inflow": inflow,
            "cash_outflow": outflow,
            "net_cashflow": round(inflow - outflow, 2)
        })

    return jsonify({"company": company_name, "timeframe": timeframe, "predictions": net_cashflow})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
