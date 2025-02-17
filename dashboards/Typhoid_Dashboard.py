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
st.set_page_config(page_title="Typhoid Data Analysis & Forecasting", layout="wide")

# 📌 Sidebar: Model Selection
st.sidebar.title("🔍 Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ["Data Insights (EDA)", "Forecasting (ARIMA) ", "Forecasting (LSTM) "])

# 📌 Load Dataset
st.title("🦠 Typhoid Data Analysis & Forecasting")
st.subheader("📂 Load and Preprocess Data")

# Load dataset from uploaded file
typhoid_data = pd.read_csv("https:/raw.githubusercontent.com//Dragonix230//predictive-analytics//refs//heads//main//data//cleaned_typhoid_data.csv")

# Drop unnecessary columns
typhoid_data = typhoid_data.drop(columns=["s.no", "name"], errors="ignore")

# Ensure 'year' column is properly formatted
if "year" in typhoid_data.columns:
    yearly_cases = typhoid_data["year"].value_counts().sort_index()
else:
    st.error("🚨 Error: 'year' column not found in dataset!")
    st.stop()

st.success("✅ Data loaded successfully!")

# 📌 Exploratory Data Analysis (EDA)
if model_choice == "Data Insights (EDA)":
    st.header("📊 Exploratory Data Analysis")
    
    # Display dataset
    st.subheader("📝 Dataset Overview")
    st.write(typhoid_data.head())
    
    st.subheader("📊 Typhoid Cases Per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=yearly_cases.index, y=yearly_cases.values, palette="viridis", ax=ax)
    ax.set_title("Total Typhoid Cases Per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cases")
    ax.set_xticklabels(yearly_cases.index, rotation=45)
    st.pyplot(fig)

    # Count cases per gender
    if "gender" in typhoid_data.columns:
        gender_cases = typhoid_data["gender"].value_counts()
        st.subheader("📊 Typhoid Cases by Gender")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=gender_cases.index, y=gender_cases.values, palette="coolwarm", ax=ax)
        ax.set_title("Typhoid Cases by Gender")
        ax.set_ylabel("Number of Cases")
        ax.set_xlabel("Gender")
        st.pyplot(fig)

    # Histogram for age distribution
    if "age" in typhoid_data.columns:
        st.subheader("📊 Age Distribution of Typhoid Patients")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(typhoid_data["age"], bins=20, kde=True, color="green", ax=ax)
        ax.set_title("Age Distribution of Typhoid Patients")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.stop()

# 📌 Aggregate Data for Forecasting
yearly_cases_series = yearly_cases.sort_index()

# 📈 ARIMA Forecast
if model_choice == "Forecasting (ARIMA)":
    st.header("📈 ARIMA Forecast for Typhoid Cases")
    
    def arima_forecast(series, steps=2):
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    
    arima_forecast_cases = arima_forecast(yearly_cases_series)
    future_years = np.arange(yearly_cases_series.index[-1] + 1, yearly_cases_series.index[-1] + 3)
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(yearly_cases_series.index, yearly_cases_series.values, label="Actual Cases", marker='o')
    ax.plot(future_years, arima_forecast_cases, label="Forecasted Cases", linestyle='dashed', marker='o', color='red')
    ax.set_title("Typhoid Cases ARIMA Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.legend()
    st.pyplot(fig)
    
    st.write("📀 ARIMA Forecast for Next 2 Years:", arima_forecast_cases.values)

# 📉 LSTM Forecast
if model_choice == "Forecasting (LSTM)":
    st.header("📉 LSTM Forecast for Typhoid Cases")
    
    cases = yearly_cases_series.values.reshape(-1, 1)
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
    
    future_year = yearly_cases_series.index[-1] + 1
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(yearly_cases_series.index, yearly_cases_series.values, label="Actual Cases", marker='o')
    ax.scatter(future_year, lstm_forecast, color='green', label="LSTM Forecast", marker='o')
    ax.set_title("Typhoid Cases LSTM Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"📀 LSTM Forecast for {future_year} Cases:", int(lstm_forecast[0][0]))
