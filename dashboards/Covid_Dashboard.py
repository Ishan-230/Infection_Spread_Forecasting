import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# Load datasets
covid_data = pd.read_csv("https://raw.githubusercontent.com//Dragonix230//predictive-analytics//refs//heads//main//data//cleaned_covid_data.csv")
testing_data = pd.read_csv("https://raw.githubusercontent.com//Dragonix230//predictive-analytics//refs//heads//main//data//cleaned_testing_data.csv")

# Convert date columns to datetime format
covid_data['datetime'] = pd.to_datetime(covid_data['date'] + ' ' + covid_data['time'], format='%Y-%m-%d %I:%M %p')
testing_data['date'] = pd.to_datetime(testing_data['date'])

# Drop duplicates and set index
covid_data = covid_data.drop_duplicates(subset=['datetime'])
covid_data.set_index('datetime', inplace=True)
covid_data = covid_data.asfreq('D')  # Set daily frequency

time_series = covid_data['confirmed'].fillna(method='ffill')

# Streamlit Dashboard
st.title("COVID-19 Cases Forecasting Dashboard")

## EDA Section
st.header("Exploratory Data Analysis (EDA)")

# Dataset Preview
st.subheader("COVID-19 Dataset Preview")
st.write(covid_data.head())

st.subheader("Testing Dataset Preview")
st.write(testing_data.head())

# Summary Statistics
st.subheader("Summary Statistics")
st.write(covid_data.describe())

# Correlation Heatmap
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(covid_data[['confirmed', 'deaths', 'cured']].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Histograms
st.subheader("Histograms of Key Features")
fig, ax = plt.subplots(1, 2, figsize=(26, 5))
covid_data[['confirmed', 'deaths', 'cured']].hist(bins=20, ax=ax[0], layout=(1,5))
testing_data.hist(bins=20, ax=ax[1], layout=(1,3))
st.pyplot(fig)

# Bar Chart
st.subheader("Bar Graph of COVID-19 Confirmed Cases")
# Filter data until May 19, 2020
covid_filtered = covid_data.loc[:'2020-04-16']

# Sum cases per date (daily aggregation)
daily_cases = covid_filtered['confirmed'].fillna(0)  # Ensure no missing values

# Plot the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
daily_cases.plot(kind='bar', color='blue', ax=ax)
ax.set_title("Daily COVID-19 Confirmed Cases (Until 2020-05-19)")
ax.set_xlabel("Date")
ax.set_ylabel("Confirmed Cases")
ax.set_xticks(range(0, len(daily_cases), max(1, len(daily_cases)//10)))  # Reduce tick labels
ax.set_xticklabels(daily_cases.index.strftime('%Y-%m-%d')[::max(1, len(daily_cases)//10)], rotation=45)  # Rotate for readability
st.pyplot(fig)

# Scatter Plot
st.subheader("Scatter Plot of Confirmed Cases Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(covid_data.index, covid_data['confirmed'], alpha=0.5, color='green')
ax.set_title("COVID-19 Confirmed Cases Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Confirmed Cases")
ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Reduce the number of date labels
plt.xticks(rotation=45)  # Rotate for readability
st.pyplot(fig)

## Time Series Plot
st.subheader("Confirmed Cases Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_series, label="Confirmed Cases", color='blue')
ax.set_title("COVID-19 Confirmed Cases Over Time")
ax.legend()
st.pyplot(fig)

# Model Selection in Sidebar
st.sidebar.header("Forecast Model")
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)

## LSTM Model
scaler = MinMaxScaler(feature_range=(0,1))
time_series_scaled = scaler.fit_transform(np.array(time_series).reshape(-1,1))

train_size = int(len(time_series_scaled) * 0.8)
train, test = time_series_scaled[:train_size], time_series_scaled[train_size:]

def create_sequences(data, time_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_train, Y_train = create_sequences(train, time_step)
X_test, Y_test = create_sequences(test, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model_lstm = Sequential([
    Input(shape=(time_step, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, Y_train, batch_size=1, epochs=2, verbose=0)

# Predict using LSTM
lstm_predictions = model_lstm.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
actual_values = time_series.iloc[-len(lstm_predictions):]

# Forecast Visualization
st.header("Forecasting Results")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_values.index, actual_values, label="Actual Data")
ax.plot(actual_values.index, lstm_predictions, label="LSTM Forecast", color="red")
ax.legend()
ax.set_title("LSTM Forecast for Confirmed Cases")
st.pyplot(fig)
