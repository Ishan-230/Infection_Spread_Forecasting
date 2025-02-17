import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 🎯 Set Page Title
st.set_page_config(page_title="Dengue Data Analysis & Forecasting", layout="wide")

# 📌 Sidebar: Model Selection
st.sidebar.title("🔍 Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ["Data Insights (EDA)", "Forecasting (ARIMA) ", "Forecasting (LSTM)","High-Risk Prediction"])

# 📌 Load Dataset
st.title("🦠 Dengue Data Analysis & Forecasting")
dengue_data = pd.read_csv("https://raw.githubusercontent.com//Dragonix230//predictive-analytics//refs//heads//main//data//cleaned_dengue_data.csv")

# Rename incorrect columns
dengue_data.rename(columns={"2024*_cases": "2024_cases", "2024*_deaths": "2024_deaths"}, inplace=True)

# Define years
years = ["2019", "2020", "2021", "2022", "2023", "2024"]
cases_columns = [f"{year}_cases" for year in years]
deaths_columns = [f"{year}_deaths" for year in years]

# Convert to numeric
dengue_data[cases_columns + deaths_columns] = dengue_data[cases_columns + deaths_columns].apply(pd.to_numeric, errors='coerce')

# 📌 State Selection
state_selected = st.sidebar.selectbox("Select a State", dengue_data['states'].unique())
state_data = dengue_data[dengue_data['states'] == state_selected]

# 📌 EDA Section
if model_choice == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    
    # Display dataset
    st.subheader("📜 Dataset Overview")
    st.write(dengue_data.head())
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(dengue_data[cases_columns + deaths_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Bar plot for total cases per year
    total_cases = dengue_data[cases_columns].sum()
    st.subheader("📊 Total Dengue Cases Per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=years, y=total_cases, palette="viridis", ax=ax)
    st.pyplot(fig)
    
    # Bar plot for total deaths per year
    total_deaths = dengue_data[deaths_columns].sum()
    st.subheader("📊 Total Dengue Deaths Per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=years, y=total_deaths, palette="Reds", ax=ax)
    st.pyplot(fig)
    
    st.stop()

# 📌 Aggregate Data for Forecasting
cases_series = pd.Series(state_data[cases_columns].values.flatten(), index=pd.to_datetime(years, format="%Y"))

# 📌 ARIMA Forecast
if model_choice == "ARIMA Forecast":
    st.header(f"📈 ARIMA Forecast for Dengue Cases in {state_selected}")
    
    def arima_forecast(series, steps=2):
        model = ARIMA(series, order=(1,1,1))  # (p,d,q) values can be tuned
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    
    arima_forecast_cases = arima_forecast(cases_series)
    
    # Generate the correct years for the forecast
    forecast_years = pd.date_range(start=cases_series.index[-1] + pd.DateOffset(years=0), periods=2, freq='Y')

    # Plot ARIMA results
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cases_series, label="Actual Cases", marker='o')
    ax.plot(forecast_years, arima_forecast_cases, linestyle='dashed', marker='o', color='red', label="Forecasted Cases")
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"📌 *ARIMA Forecast for 2025-2026 in {state_selected}:*", arima_forecast_cases.values)

# 📌 LSTM Forecast
if model_choice == "LSTM Forecast":
    st.header(f"📉 LSTM Forecast for Dengue Cases in {state_selected}")
    
    # Prepare data for LSTM
    cases = cases_series.values.reshape(-1, 1)
    
    # Create sequences for training
    X, y = [], []
    seq_length = 3  # Using past 3 years to predict the next year

    for i in range(len(cases) - seq_length):
        X.append(cases[i:i+seq_length])
        y.append(cases[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    
    # Build LSTM Model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    
    # Predict next 2 years (2025, 2026)
    predictions = []
    input_data = cases[-seq_length:].reshape(1, seq_length, 1)  # Start with last 3 known cases

    for _ in range(2):  # Predict for the next 2 years
        next_year_forecast = model.predict(input_data)[0][0]
        predictions.append(next_year_forecast)
        input_data = np.append(input_data[:, 1:, :], [[[next_year_forecast]]], axis=1)  # Shift window
    
    # Generate forecast years
    forecast_years = [cases_series.index[-1] + pd.DateOffset(years=i) for i in range(1, 3)]
    
    # Plot LSTM results
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cases_series, label="Actual Cases", marker='o')
    ax.scatter(forecast_years, predictions, color='green', label="LSTM Forecast", marker='o')
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"📌 *LSTM Forecast for 2025 & 2026 Cases in {state_selected}:*")
    st.write(f"📅 Predicted Cases for 2025: **{int(predictions[0])}**")
    st.write(f"📅 Predicted Cases for 2026: **{int(predictions[1])}**")

    # 📌 High-Risk State Prediction
elif model_choice == "High-Risk Prediction":
    st.header("🚨 Predicting Most Affected States")

    # Encode categorical 'states' column
    label_encoder = LabelEncoder()
    dengue_data['state_encoded'] = label_encoder.fit_transform(dengue_data['states'])

    # Define features (past cases) and target (latest cases)
    predictors = ['state_encoded'] + cases_columns[:-1]  # Use past years' data for prediction
    target_column = cases_columns[-1]  # Latest available year for prediction

    X = dengue_data[predictors]
    y = dengue_data[target_column]

    # Train the model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X, y)

    # Predict risk scores
    dengue_data['risk_score'] = model_rf.predict(X)

    # Identify high-risk states
    top_risk_states = dengue_data[['states', 'risk_score']].groupby('states').mean().sort_values(by='risk_score', ascending=False).head(5)

    st.subheader("🔥 Top 5 High-Risk States")
    st.write(top_risk_states)

    # Plot risk scores
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_risk_states.index, y=top_risk_states['risk_score'], palette="Reds", ax=ax)
    ax.set_ylabel("Predicted Risk Score")
    ax.set_xlabel("States")
    ax.set_title("Top 5 High-Risk States for Dengue")
    st.pyplot(fig)
