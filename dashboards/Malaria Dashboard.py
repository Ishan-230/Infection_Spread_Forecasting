import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 🎯 Set Page Title
st.set_page_config(page_title="Malaria Data Analysis & Forecasting", layout="wide")

# 📌 Sidebar: Model Selection
st.sidebar.title("🔍 Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ["Exploratory Data Analysis (EDA)", "ARIMA Forecast", "LSTM Forecast"])

# 📌 Load Dataset
st.title("🦟 Malaria Data Analysis & Forecasting")
st.subheader("📂 Load and Preprocess Data")
malaria_data = pd.read_csv("https://raw.githubusercontent.com//Dragonix230//predictive-analytics//refs//heads//main//data//cleaned_malaria_data.csv")

# Convert data to time series format
years = ["Cases_2020", "Cases_2021", "Cases_2022", "Cases_2023", "Cases_2024_Upto_Nov"]
yearly_cases = malaria_data[years].sum()
yearly_cases.index = [2020, 2021, 2022, 2023, 2024]

st.success("✅ Data loaded successfully!")

# 📌 Exploratory Data Analysis (EDA)
if model_choice == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    
    # Display dataset
    st.subheader("📝 Dataset Overview")
    st.write(malaria_data.head())
    
    # Bar Plot
    st.subheader("📊 Malaria Cases Per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=yearly_cases.index, y=yearly_cases.values, palette="viridis", ax=ax)
    ax.set_title("Total Malaria Cases Per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cases")
    ax.set_xticklabels(yearly_cases.index, rotation=45)
    st.pyplot(fig)
    
    # Heatmap for correlations
    st.subheader("🌬 Heatmap of Feature Correlations")
    numeric_data = malaria_data.select_dtypes(include=['number'])  # Select only numeric columns
    if numeric_data.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("⚠ Not enough numeric columns for correlation heatmap.")
    
    # Histogram
    st.subheader("📊 Histogram of Malaria Cases")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(yearly_cases, bins=10, kde=True, ax=ax, color='green')
    ax.set_title("Distribution of Malaria Cases Over Years")
    ax.set_xlabel("Cases")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    # Scatter Plot
    st.subheader("📌 Scatter Plot of Malaria Cases")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=yearly_cases.index, y=yearly_cases.values, marker='o', color='red', ax=ax)
    ax.set_title("Malaria Cases Scatter Plot")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cases")
    st.pyplot(fig)
    
    st.stop()

# 📌 ARIMA Forecast
if model_choice == "ARIMA Forecast":
    st.header("📈 ARIMA Forecast for Malaria Cases")
    
    def arima_forecast(series, steps=2):
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    arima_forecast_cases = arima_forecast(yearly_cases)
    future_years = np.arange(2025, 2027)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(yearly_cases.index, yearly_cases.values, label="Actual Cases", marker='o')
    ax.plot(future_years, arima_forecast_cases, linestyle='dashed', marker='o', color='red', label="Forecasted Cases")
    ax.set_title("Malaria Cases ARIMA Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.legend()
    st.pyplot(fig)
    
    st.write("📌 ARIMA Forecast for Next 2 Years:", arima_forecast_cases.values)

# 📌 LSTM Forecast
if model_choice == "LSTM Forecast":
    st.header("📉 LSTM Forecast for Malaria Cases")

    cases = yearly_cases.values.reshape(-1, 1)
    X, y = [], []
    for i in range(len(cases) - 3):
        X.append(cases[i:i+3])
        y.append(cases[i+3])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    input_data = cases[-3:].reshape(1, 3, 1)
    lstm_forecast = model.predict(input_data)
    
    future_year = 2025
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(yearly_cases.index, yearly_cases.values, label="Actual Cases", marker='o')
    ax.scatter(future_year, lstm_forecast, color='green', label="LSTM Forecast", marker='o')
    ax.set_title("Malaria Cases LSTM Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"📌 LSTM Forecast for {future_year} Cases:", int(lstm_forecast[0][0]))
