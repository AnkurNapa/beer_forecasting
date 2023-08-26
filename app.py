import streamlit as st
# import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot

# apply seaborn aesthetics to all matplotlib plots
sns.set_theme(style="darkgrid")
import statsmodels.api as sm
from pylab import rcParams
from datetime import datetime
import pickle

import warnings
warnings.filterwarnings('ignore')
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', InterpolationWarning)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import re
import joblib

#Using Complete width
st.set_page_config(layout="wide")

# Code Starting
name = "Beer Sales Forecasting Case Study"
st.text(name)

# Title and Subheader
st.title("Forecasting App")
st.subheader("Beer Forecating App with Streamlit ")


# EDA
my_dataset = 'Forecasting_case_study.csv'

# To Improve speed and cache data
@st.cache_data(persist=True)
def explore_data(dataset):
	data = pd.read_csv('Forecasting_case_study.csv')
	data['Type'] = data['Product Name'].map(lambda x: x.split(' ')[1])
	data['Volume'] = data['Product Name'].map(lambda x: ' '.join(x.split(' ')[-2:]))
	data['Variant'] = data['Product Name'].map(lambda x: ' '.join(x.split(' ')[2:-2]))
	data['Volume'] = data['Volume'].map(lambda x: 330 if x == '330 ml' else (500 if x == '500 ml' else 473))
	data['Final_Vol'] = data['Volume']*data['sales']*0.00001
	data['date'] = data['date'].map(lambda x: datetime.strptime(x, '%d-%m-%Y'))
	data['date'] = pd.to_datetime(data['date'])

	data['year'] = data['date'].map(lambda x:x.year)
	data['month'] = data['date'].map(lambda x:x.month)
	data['day'] = data['date'].map(lambda x:int(x.strftime('%j')))
	data['weekday'] = data['date'].map(lambda x:x.weekday())
	data['year-month'] = data.apply(lambda x:str(x['year']) + '-' + str(x['month']), axis=1)
	return data 

# Our Dataset
data = explore_data(my_dataset)

# Show Entire Dataframe
if st.checkbox("Show All DataFrame"):
	st.dataframe(data)

city_filter = st.sidebar.selectbox('Select Category', data['Store_city'].unique())
product_filter = st.sidebar.selectbox('Select Category', data['Product Name'].unique())

if st.checkbox('Exploratory Data Analysis'):
    # City Level Chart
    store_sales = data.groupby(['Store_city'], as_index=False)['Final_Vol'].sum()
    store_sales.sort_values('Final_Vol', inplace=True, ascending=False)
    plt.figure(figsize=(10,4))
    plt.title('Volume Consumption across Cities')
    sns.barplot(data=store_sales,x='Store_city',y='Final_Vol')
    plt.xticks(rotation=45);
    st.pyplot(plt)


    # Best Selling Product in city


    product_sales = data[data['Store_city'] == city_filter].groupby(['Product Name'], as_index=False)['Final_Vol'].sum()
    product_sales.sort_values('Final_Vol', inplace=True, ascending=False)


    plt.title('Total Volume Consumed across ' + city_filter + ' for various products')
    plt.figure(figsize=(30,7))
    sns.barplot(data=product_sales,x='Product Name',y='Final_Vol')
    plt.xticks(rotation=90);
    st.pyplot(plt)


    # City - Product Distribution
    filtered_df = data[(data['Store_city'] == city_filter) & (data['Product Name'] == product_filter)]



    combine_fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Bulbasaur
    sns.boxplot(x="weekday", y="Final_Vol", data= filtered_df, ax=axes[0]);
    axes[0].set_title('Sales per week day')

    # Charmander
    sns.boxplot(x="month", y="Final_Vol", data= filtered_df, ax=axes[1]);
    axes[1].set_title("Sales per month.")

    # Squirtle
    sns.lineplot(data= filtered_df,  x='month',  y='Final_Vol', hue='year', legend='full', palette='Dark2', ax=axes[2],)
    axes[2].set_title('Year Seasonlity Plot')

    wide_fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
    sns.boxplot(x="year-month", y="Final_Vol", data= filtered_df);
    axes.set_title('Sales per Year-Month')
    plt.xticks(rotation=45);

    st.pyplot(combine_fig)
    st.pyplot(wide_fig)

filtered_df = data[(data['Store_city'] == city_filter) & (data['Product Name'] == product_filter)][['Final_Vol','date']]
filtered_df.set_index('date', inplace=True, drop=True)

st.subheader("Time Series Exploration ")
decomposition = sm.tsa.seasonal_decompose(filtered_df, model = 'additive', period=365)                                    
fig = decomposition.plot()
st.pyplot(plt)

pickle_file_path = 'arima/' + city_filter + '_' + product_filter + '_arima.pkl'
model = joblib.load(pickle_file_path)

arima_result = model.fit()
mae_arima = np.mean(np.abs(arima_result.resid))

arima_pred = arima_result.get_forecast(steps=90)
arima_mean = pd.DataFrame(arima_pred.predicted_mean)


sarima_02_model = SARIMAX(filtered_df, order=(6, 1, 1), seasonal_order=(6, 1, 0, 7))
sarima_02_results = sarima_02_model.fit(disp=False)

# Calculate the mean absolute error from residuals
mae_sarima = np.mean(np.abs(sarima_02_results.resid))



# Create SARIMA mean forecast
sarima_02_pred = sarima_02_results.get_forecast(steps=90)
sarima_02_mean = pd.DataFrame(sarima_02_pred.predicted_mean)


st.text('MAE Arima Model: %.3f' % mae_arima)
st.text('MAE Sarima Model: %.3f' % mae_sarima)


predicted_final_data = pd.merge(arima_mean, sarima_02_mean, left_index=True, right_index=True)
predicted_final_data.columns = ['Arima Prediction', 'Sarima Prediction']
st.dataframe(predicted_final_data)

dates = filtered_df.index
# Plot mean ARIMA and SARIMA predictions and observed
plt.figure()
plt.title("Comparing Forecasting 90 days ahead - ARIMA vs SARIMA", size =16)
plt.plot(filtered_df['2017':], label='observed')
plt.plot(arima_mean.index, arima_mean, label='ARIMA')
plt.plot(sarima_02_mean.index, sarima_02_mean, label='SARIMA')
plt.legend()
plt.show()

st.pyplot(plt)

