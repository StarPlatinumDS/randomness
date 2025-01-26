import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from docutils.nodes import title
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

from statsmodels.tsa.vector_ar.var_model import forecast
from sympy.physics.units import frequency


#Load and preprocess data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error('Unsupported file type. Please upload .csv or .xlsx')
            return None
        return data
    except Exception as e:
        st.error(f'Error reading file: {e}')
        return None


#Preprocess time series data
def preprocess_data(data, date_column, value_column):
    try:
        data[date_column] = pd.to_datetime(data[date_column])
        data = data[[date_column, value_column]].dropna()
        data = data.set_index(date_column).sort_index()
        return data
    except Exception as e:
        st.error(f'Error preprocessing file: {e}')
        return None


#Visualize time series
def plot_time_series(data, value_column):
    st.subheader('Time Series Visualization')
    fig, ax = plt.subplots(figsize=(12, 6))
    data[value_column].plot(ax=ax, color='blue', title='Time Series Plot')
    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    st.pyplot(fig)


#Perform rolling average
def plot_rolling_average(data, value_column, window):
    st.subheader(f'Rolling Average (Window = {window} Days)')
    fig, ax = plt.subplots(figsize=(12, 6))
    data[value_column].rolling(window=window).mean().plot(ax=ax, color='red', label='Rolling Average')
    data[value_column].plot(ax=ax, alpha=0.5, label='Original')
    ax.set_title(f'Rolling Average vs. Original Data')
    ax.legend()
    st.pyplot(fig)


#Decompose the time series
def decompose_time_series(data, value_column, frequency):
    st.subheader('Time Series Decomposition')
    decomposition = seasonal_decompose(data[value_column], period=frequency)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)


#Train ARIMA model and forecast
def arima_forecast(data, value_column, p, d, q, forecast_steps):
    st.subheader('ARIMA Model Forecast')
    try:
        model = ARIMA(data[value_column], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)

        #Plot forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        data[value_column].plot(ax=ax, label='Original Data', color='blue')
        forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='D')[1:]
        ax.plot(forecast_index, forecast, label='Forecast', color='green')
        ax.set_title('ARIMA Model Forecast')
        ax.legend()
        st.pyplot(fig)

        #Metrics
        st.write("### Model Summary")
        st.text(model_fit.summary())
    except Exception as e:
        st.error(f'Error in ARIMA Model: {e}')


#Streamlit Dashboard
def main():
    st.title('Time Series Data Visualization')
    #File upload
    uploaded_file = st.file_uploader('Upload Time Series Data (CSV or Excel)', type=['csv', 'xlsx'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write('### Uploaded Data')
            st.write(data.head())

            #Select date and value columns
            date_column = st.selectbox('Select Date Column', data.columns)
            value_column = st.selectbox('Select Value Column', data.columns)

            if st.button('Preprocess Data'):
                ts_data = preprocess_data(data, date_column, value_column)
                if ts_data is not None:
                    st.write('### Preprocessed Data')
                    st.write(ts_data.head())

                    #Visualize
                    plot_time_series(ts_data, value_column)

                    #Rolling Average
                    window = st.slider('Select Rolling Window (Days)', 1, 7, 30)
                    plot_rolling_average(ts_data, value_column, window)

                    #Decomposition
                    frequency = st.number_input('Seasonal Frequency (7 for weekly data)', min_value=1, max_value=7)
                    decompose_time_series(ts_data, value_column, frequency)

                    #ARIMA forecsat
                    st.write('ARIMA Model Configuration')
                    p = st.number_input('ARIMA (p): Autoregressive Order', min_value=0, max_value=1)
                    d = st.number_input('ARIMA (d): Differencing Order', min_value=0, max_value=1)
                    q = st.number_input('ARIMA (q): Moving Average Order', min_value=0, max_value=1)
                    forecast_steps = st.number_input('Forecast Steps (Days)', min_value=1, value=30)

                    if st.button('Train ARIMA Model and Forecast'):
                        arima_forecast(ts_data, value_column, p, d, q, forecast_steps)


if __name__ == '__main__':
    main()