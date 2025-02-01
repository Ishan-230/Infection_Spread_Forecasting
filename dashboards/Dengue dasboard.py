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
st.set_page_config(page_title="Dengue Data Analysis & Forecasting", layout="wide")

# 📌 Sidebar: Model Selection
st.sidebar.title("🔍 Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ["Exploratory Data Analysis (EDA)", "ARIMA Forecast", "LSTM Forecast"])

# 📌 Load Dataset
st.title("🦠 Dengue Data Analysis & Forecasting")
dengue_data = pd.read_csv("cleaned_dengue_data.csv")

# Rename incorrect columns
dengue_data.rename(columns={"2024*_cases": "2024_cases", "2024*_deaths": "2024_deaths"}, inplace=True)

# Define years
years = ["2019", "2020", "2021", "2022", "2023", "2024"]
cases_columns = [f"{year}_cases" for year in years]
deaths_columns = [f"{year}_deaths" for year in years]

# Check if columns exist
missing_cols = [col for col in (cases_columns + deaths_columns) if col not in dengue_data.columns]
if missing_cols:
    st.error(f"🚨 Missing columns in dataset: {missing_cols}")
    st.stop()

# Convert to numeric
dengue_data[cases_columns + deaths_columns] = dengue_data[cases_columns + deaths_columns].apply(pd.to_numeric, errors='coerce')

# 📌 EDA Section
if model_choice == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")

    # Display dataset
    st.subheader("📜 Dataset Overview")
    st.write(dengue_data.head())

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    numeric_cols = dengue_data.select_dtypes(include=['number'])
    correlation_matrix = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Total cases and deaths per year
    total_cases = dengue_data[cases_columns].sum()
    total_deaths = dengue_data[deaths_columns].sum()

    # Bar plot for total cases per year
    st.subheader("📊 Total Dengue Cases Per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=years, y=total_cases, palette="viridis", ax=ax)
    ax.set_title("Total Dengue Cases Per Year")
    ax.set_ylabel("Total Cases")
    ax.set_xlabel("Year")
    st.pyplot(fig)

    # Bar plot for total deaths per year
    st.subheader("📊 Total Dengue Deaths Per Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=years, y=total_deaths, palette="Reds", ax=ax)
    ax.set_title("Total Dengue Deaths Per Year")
    ax.set_ylabel("Total Deaths")
    ax.set_xlabel("Year")
    st.pyplot(fig)

    # Histogram for 2023 cases
    st.subheader("📊 Distribution of Dengue Cases in 2023")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(dengue_data["2023_cases"], bins=30, kde=True, color="purple", ax=ax)
    ax.set_title("Distribution of Dengue Cases in 2023")
    ax.set_xlabel("Number of Cases")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Scatter plot for 2023 cases vs deaths
    st.subheader("📊 Dengue Deaths vs Cases in 2023")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=dengue_data["2023_cases"], y=dengue_data["2023_deaths"], color="blue", ax=ax)
    ax.set_title("Dengue Deaths vs Cases in 2023")
    ax.set_xlabel("Cases")
    ax.set_ylabel("Deaths")
    st.pyplot(fig)

    st.stop()

# 📌 Aggregate Data for Forecasting
total_cases = dengue_data[cases_columns].sum()
cases_series = pd.Series(total_cases.values, index=pd.to_datetime(years, format="%Y"))

# 📌 ARIMA Forecast
if model_choice == "ARIMA Forecast":
    st.header("📈 ARIMA Forecast for Dengue Cases")

    def arima_forecast(series, steps=2):
        model = ARIMA(series, order=(1,1,1))  # (p,d,q) values can be tuned
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    arima_forecast_cases = arima_forecast(cases_series)

    # Plot ARIMA results
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cases_series, label="Actual Cases", marker='o')
    ax.plot(pd.date_range(start=cases_series.index[-1], periods=3, freq='Y')[1:], 
            arima_forecast_cases, label="Forecasted Cases", linestyle='dashed', marker='o', color='red')
    ax.set_title("Dengue Cases ARIMA Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.legend()
    st.pyplot(fig)

    # Print Forecast
    st.write("📌 **ARIMA Forecast for Next 2 Years:**", arima_forecast_cases.values)

# 📌 LSTM Forecast
if model_choice == "LSTM Forecast":
    st.header("📉 LSTM Forecast for Dengue Cases")

    # Prepare data for LSTM
    cases = cases_series.values.reshape(-1, 1)

    # Create sequences (use past 3 years to predict next year)
    X, y = [], []
    for i in range(len(cases) - 3):
        X.append(cases[i:i+3])
        y.append(cases[i+3])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Build LSTM Model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train Model
    model.fit(X, y, epochs=200, verbose=0)

    # Predict next year
    input_data = cases[-3:].reshape(1, 3, 1)  # Last 3 years
    lstm_forecast = model.predict(input_data)

    # Plot LSTM results
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cases_series, label="Actual Cases", marker='o')
    ax.scatter(pd.to_datetime("2025"), lstm_forecast, color='green', label="LSTM Forecast", marker='o')
    ax.set_title("Dengue Cases LSTM Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Cases")
    ax.legend()
    st.pyplot(fig)

    # Print Forecast
    st.write("📌 **LSTM Forecast for 2025 Cases:**", lstm_forecast[0][0])
