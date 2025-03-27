from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel

app = FastAPI()

# API URL for fetching payment data
API_URL = "http://192.168.1.2:9999/api/payment_data"


# Request model
class PredictionRequest(BaseModel):
    companyName: str
    timeframe: str = "daily"  # hourly, daily, weekly


# Function to fetch and filter data by company
def get_company_data(company_name):
    response = requests.get(API_URL)
    if response.status_code != 200:
        return None

    data = response.json()
    df = pd.DataFrame(data)

    if "creationDateTime" not in df.columns:
        raise KeyError("Missing 'creationDateTime' column in API response")

    # Convert to datetime
    df["creationDateTime"] = pd.to_datetime(df["creationDateTime"].apply(lambda x: "-".join(map(str, x[:3]))))

    # Filter transactions where the company is either debtor (outflow) or creditor (inflow)
    inflow_data = df[df["creditorName"] == company_name]
    outflow_data = df[df["debtorName"] == company_name]

    return inflow_data, outflow_data if not inflow_data.empty or not outflow_data.empty else None


# Function to train LSTM model
def train_lstm(data):
    if data.empty:
        return None, None, None  # Return None if no training data

    scaler = MinMaxScaler(feature_range=(0, 1))
    data["amount"] = scaler.fit_transform(data["amount"].values.reshape(-1, 1))

    # Prepare time series data
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data["amount"].iloc[i])
        y.append(data["amount"].iloc[i + 1])

    if len(X) == 0:
        return None, None, None  # No valid data for training

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


# Function to predict future cash flow (Inflow, Outflow, Net Cash Flow)
def predict_cashflow(model, scaler, last_known_value, timeframe="daily", steps=7):
    if model is None:
        return [{"datetime": str(datetime.datetime.now()), "cash_inflow": 0, "cash_outflow": 0, "net_cashflow": 0}]

    predictions = []
    interval = {"hourly": "hours", "weekly": "days"}.get(timeframe, "days")  # Weekly means next 7 days, daily is 1 day

    steps = 7 if timeframe == "weekly" else 1  # Predict next 1 day for "daily", next 7 days for "weekly"

    future_dates = [datetime.datetime.now() + datetime.timedelta(**{interval: i}) for i in range(1, steps + 1)]

    for date in future_dates:
        predicted_value = model.predict(last_known_value.reshape(1, 1, 1))[0][0]
        last_known_value = np.array([[predicted_value]])  # Update for next prediction

        cashflow = round(scaler.inverse_transform([[predicted_value]])[0][0], 2)
        predictions.append({
            "datetime": str(date),
            "cash_inflow": cashflow if np.random.rand() > 0.5 else 0,
            "cash_outflow": cashflow if np.random.rand() > 0.5 else 0,
            "net_cashflow": round(cashflow if np.random.rand() > 0.5 else -cashflow, 2)
        })

    return predictions


# Function to predict by network (ACH, FED, etc.)
def predict_by_network(df, company_name, timeframe, steps=7):
    networks = df["debtorNetworkType"].unique()
    network_predictions = {}

    for network in networks:
        network_data = df[df["debtorNetworkType"] == network]
        inflow_data, outflow_data = network_data[network_data["creditorName"] == company_name], network_data[
            network_data["debtorName"] == company_name]

        model_inflow, scaler_inflow, last_inflow = train_lstm(inflow_data)
        model_outflow, scaler_outflow, last_outflow = train_lstm(outflow_data)

        inflow_predictions = predict_cashflow(model_inflow, scaler_inflow, last_inflow, timeframe, steps)
        outflow_predictions = predict_cashflow(model_outflow, scaler_outflow, last_outflow, timeframe, steps)

        network_predictions[network] = {
            "cash_inflow": inflow_predictions,
            "cash_outflow": outflow_predictions
        }

    return network_predictions


# FastAPI POST Endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
    company_name = request.companyName
    timeframe = request.timeframe  # hourly, daily, weekly

    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required")

    inflow_data, outflow_data = get_company_data(company_name)
    if inflow_data is None and outflow_data is None:
        raise HTTPException(status_code=404, detail=f"No data found for {company_name}")

    model_inflow, scaler_inflow, last_inflow = train_lstm(inflow_data)
    model_outflow, scaler_outflow, last_outflow = train_lstm(outflow_data)

    inflow_predictions = predict_cashflow(model_inflow, scaler_inflow, last_inflow, timeframe, steps=7)
    outflow_predictions = predict_cashflow(model_outflow, scaler_outflow, last_outflow, timeframe, steps=7)

    # Calculate net cash flow
    net_cashflow_predictions = []
    for inflow, outflow in zip(inflow_predictions, outflow_predictions):
        net_cashflow_predictions.append({
            "datetime": inflow["datetime"],
            "cash_inflow": inflow["cash_inflow"],
            "cash_outflow": outflow["cash_outflow"],
            "net_cashflow": inflow["cash_inflow"] - outflow["cash_outflow"]
        })

    # Network-based predictions
    network_predictions = predict_by_network(pd.concat([inflow_data, outflow_data]), company_name, timeframe, steps=7)

    return {
        "company": company_name,
        "timeframe": timeframe,
        "predictions": {
            "cash_inflow": inflow_predictions,
            "cash_outflow": outflow_predictions,
            "net_cashflow": net_cashflow_predictions,
            "network_predictions": network_predictions
        }
    }


# Run with Uvicorn
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
