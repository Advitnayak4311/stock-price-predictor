# ----------------------------
# STOCK PRICE PREDICTOR
# ----------------------------

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1 – Download Stock Data
print("Downloading stock data for Apple (AAPL)...")
data = yf.download("AAPL", start="2023-01-01", end="2025-01-01")
print(data.head())

# Step 2 – Prepare Dataset
data['Next_Close'] = data['Close'].shift(-1)  # Next day's close
data = data.dropna()

X = data[['Close']]  # Features
y = data['Next_Close']  # Target

# Step 3 – Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4 – Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5 – Make Predictions
predictions = model.predict(X_test)

# Step 6 – Visualize Actual vs Predicted Prices
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='orange')
plt.title("Apple Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Step 7 – Test with a Custom Price
today_price = float(input("Enter today's closing price to predict tomorrow's price: "))
predicted_price = model.predict([[today_price]])
print(f"Predicted next day price: ${predicted_price[0]:.2f}")
