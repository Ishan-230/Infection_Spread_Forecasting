import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit Page Configuration
st.set_page_config(page_title="Malaria Data Analysis & Forecasting", layout="wide")
st.title("🦟 Malaria Data Analysis & Forecasting")

# Load Dataset
file_path = r"https://raw.githubusercontent.com//Dragonix230//predictive-analytics//refs//heads//main//data//cleaned_malaria_data.csv"
malaria_data = pd.read_csv(file_path)
malaria_data.rename(columns={'State/UT': 'state'}, inplace=True)

# Define Years & Columns
years = [2020, 2021, 2022, 2023, 2024]
cases_columns = ["Cases_2020", "Cases_2021", "Cases_2022", "Cases_2023", "Cases_2024_Upto_Nov"]

# Sidebar Options
st.sidebar.title("🔍 Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ["Data Insights (EDA)", "Forecasting (ARIMA)", "Forecasting (LSTM)", "High-Risk Prediction"])
state_selected = st.sidebar.selectbox("Select a State", malaria_data['state'].unique())

# Prepare Data
yearly_cases = malaria_data[cases_columns].sum()
yearly_cases.index = years
state_yearly_cases = malaria_data[cases_columns].sum()
state_yearly_cases.index = years
state_data = malaria_data[malaria_data['state'] == state_selected]

def plot_time_series(data, title="Malaria Cases Over Years"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data.values, marker='o', label='Reported Cases', color='blue')
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cases")
    ax.legend()
    st.pyplot(fig)

# 📌 Exploratory Data Analysis (EDA)
if model_choice == "Data Insights (EDA)":
    st.header("📊 Exploratory Data Analysis")
    st.write(malaria_data.head())
    plot_time_series(yearly_cases, "Total Malaria Cases Per Year")
    
    # Correlation Heatmap
    numeric_data = malaria_data.select_dtypes(include=['number'])
    if numeric_data.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

# 📌 ARIMA Forecasting
elif model_choice == "Forecasting (ARIMA)":
    st.header(f"📈 ARIMA Forecast for {state_selected}")

    state_yearly_cases = state_data[cases_columns].sum()
    state_yearly_cases.index = years

    if state_yearly_cases.sum() == 0:
        st.warning("No sufficient data available for this state.")
    else:
        model_arima = ARIMA(state_yearly_cases, order=(1, 1, 1))
        model_arima_fit = model_arima.fit()
        forecast_arima = model_arima_fit.forecast(steps=2)
        future_years = np.arange(2025, 2027)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(state_yearly_cases.index, state_yearly_cases.values, label="Actual Cases", marker='o')
        ax.plot(future_years, forecast_arima, linestyle='dashed', marker='o', color='red', label="ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)
        st.write(f"📌 ARIMA Forecast for {state_selected}:", forecast_arima.values)

# 📌 LSTM Forecasting
elif model_choice == "Forecasting (LSTM)":
    st.header("📉 LSTM Forecast for Malaria Cases (2025 & 2026)")

    cases = yearly_cases.values.reshape(-1, 1)

    # Prepare Training Data: Each input sequence (X) has 3 years, and output (y) has 2 future years
    X, y = [], []
    for i in range(len(cases) - 4):  # Now predicting 2 years ahead
        X.append(cases[i:i+3])
        y.append(cases[i+3:i+5])  # Predict next 2 years

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # LSTM Model with 2 Output Neurons (for 2025 & 2026)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 1)),
        Dense(2)  # Output two values (2025 & 2026)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    # Use last 3 years to predict next 2 years (2025 & 2026)
    input_data = cases[-3:].reshape(1, 3, 1)
    lstm_forecast = model.predict(input_data)[0]

    # Plot Forecast
    future_years = [2025, 2026]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly_cases.index, yearly_cases.values, label="Actual Cases", marker='o')
    ax.scatter(future_years, lstm_forecast, color='green', label="LSTM Forecast", marker='o')
    ax.legend()
    st.pyplot(fig)

    # Display Predictions
    st.write(f"📌 LSTM Forecast:")
    st.write(f"📅 Predicted Cases for 2025: **{int(lstm_forecast[0])}**")
    st.write(f"📅 Predicted Cases for 2026: **{int(lstm_forecast[1])}**")

# 📌 High-Risk State Prediction
elif model_choice == "High-Risk Prediction":
    st.header("🚨 Predicting Most Affected States for Malaria")

    # Encode state names as numerical values
    malaria_data['state_encoded'] = LabelEncoder().fit_transform(malaria_data['state'])
    predictors = ['state_encoded'] + cases_columns

    # Define Features (X) and Target Variable (y)
    X = malaria_data[predictors]
    y = malaria_data['Cases_2024_Upto_Nov']  # Using 2024 up to Nov data

    # Train Random Forest Model
    model_rf = RandomForestRegressor(random_state=42, n_estimators=100)
    model_rf.fit(X, y)

    # Predict risk scores for states
    malaria_data['risk_score'] = model_rf.predict(X)

    # Get top 5 high-risk states
    top_risk_states = malaria_data.groupby('state')['risk_score'].mean().sort_values(ascending=False).head(5)

    # Display top high-risk states as a table
    st.write("### 🔥 Top 5 High-Risk States for Malaria (2024)")
    st.write(top_risk_states)

    # 📊 Visualization - Bar Chart
    st.subheader("📉 High-Risk States for Malaria")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_risk_states.index, y=top_risk_states.values, palette="Reds", ax=ax)
    ax.set_ylabel("Predicted Risk Score")
    ax.set_xlabel("State")
    ax.set_title("Top 5 High-Risk States for Malaria")
    st.pyplot(fig)
