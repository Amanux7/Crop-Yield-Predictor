import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utils import predict_yield

# Paths
DATA_PATH = 'data/yield_df.csv'
MODEL_PATH = 'models/crop_yield_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
POLY_PATH = 'models/poly_features.pkl'

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    poly = joblib.load(POLY_PATH)
    return model, scaler, poly

model, scaler, poly = load_artifacts()

# Load historical data for dashboard
@st.cache_data
def load_historical_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df['Item'].isin(['Wheat', 'Rice, paddy', 'Maize'])]
    return df

historical_df = load_historical_data()

# Unique values for dropdowns
unique_countries = sorted(historical_df['Area'].unique())
unique_crops = sorted(historical_df['Item'].unique())

# App title and sidebar
st.title('Crop Yield Prediction System')
st.sidebar.header('Navigation')
page = st.sidebar.radio('Select Page', ['Single Prediction', 'Batch Prediction', 'Dashboard'])

if page == 'Single Prediction':
    st.header('Predict Yield for a Single Input')
    st.write('Enter details below (aligned with personas like Raj for quick forecasts).')
    
    # Inputs
    area = st.selectbox('Country/Area', unique_countries)
    item = st.selectbox('Crop', unique_crops)
    year = st.number_input('Year', min_value=1990, max_value=2050, value=2023)
    rainfall = st.number_input('Average Rainfall (mm/year)', min_value=0.0, value=1000.0)
    pesticides = st.number_input('Pesticides (tonnes)', min_value=0.0, value=100.0)
    temp = st.number_input('Average Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
    
    if st.button('Predict Yield'):
        # Prepare input
        input_data = {
            'Area': area,
            'Item': item,
            'Year': year,
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides,
            'avg_temp': temp
        }
        try:
            yield_pred, lower_ci, upper_ci = predict_yield(input_data, model, scaler, poly)
            st.success(f'Predicted Yield: {yield_pred:.2f} tons/ha (95% CI: {lower_ci:.2f} - {upper_ci:.2f})')
            
            # Alert for low yield
            avg_yield = historical_df['hg/ha_yield'].mean() / 10000  # Convert to tons/ha
            if yield_pred < 0.9 * avg_yield:
                st.warning('Alert: Predicted yield is below average—consider resource optimization.')
        except ValueError as e:
            st.error(f"Error: {e}")

elif page == 'Batch Prediction':
    st.header('Batch Prediction via CSV Upload')
    st.write('Upload a CSV with columns: Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.')
    
    uploaded_file = st.file_uploader('Choose CSV file', type='csv')
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        if all(col in batch_df.columns for col in required_cols):
            try:
                preds, lower_ci, upper_ci = predict_yield(batch_df, model, scaler, poly)
                result_df = batch_df.copy()
                result_df['Predicted_Yield_tons_ha'] = preds
                result_df['CI_Lower'] = lower_ci
                result_df['CI_Upper'] = upper_ci
                
                st.dataframe(result_df)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Predictions CSV', csv, 'predictions.csv', 'text/csv')
                
                # Alerts
                low_yield_count = (result_df['Predicted_Yield_tons_ha'] < 0.9 * (historical_df['hg/ha_yield'].mean() / 10000)).sum()
                if low_yield_count > 0:
                    st.warning(f'Alert: {low_yield_count} entries have below-average yields.')
            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.error(f'CSV missing required columns: {required_cols}')

elif page == 'Dashboard':
    st.header('Dashboard: Visualizations and Insights')
    st.write('Explore historical trends and correlations.')
    
    # Filters
    selected_crop = st.selectbox('Select Crop for Charts', unique_crops)
    filtered_df = historical_df[historical_df['Item'] == selected_crop]
    
    # Chart 1: Yield over Years
    st.subheader('Yield Trend Over Years')
    fig1, ax1 = plt.subplots()
    filtered_df.groupby('Year')['hg/ha_yield'].mean().plot(ax=ax1)
    ax1.set_ylabel('Average Yield (hg/ha)')
    st.pyplot(fig1)
    
    # Chart 2: Yield vs Temperature
    st.subheader('Yield vs Average Temperature')
    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered_df['avg_temp'], filtered_df['hg/ha_yield'])
    ax2.set_xlabel('Avg Temp (°C)')
    ax2.set_ylabel('Yield (hg/ha)')
    st.pyplot(fig2)
    
    # Chart 3: Top Countries by Yield
    st.subheader('Top 10 Countries by Average Yield')
    top_countries = filtered_df.groupby('Area')['hg/ha_yield'].mean().nlargest(10)
    fig3, ax3 = plt.subplots()
    top_countries.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Average Yield (hg/ha)')
    st.pyplot(fig3)
    
    # Table of yields by country
    st.subheader('Yields by Country (Table View)')
    country_yields = filtered_df.groupby('Area')['hg/ha_yield'].mean().reset_index()
    st.dataframe(country_yields)
    
    st.info('Note: Interactive maps can be added in Phase 3 with folium.')

# Footer
st.sidebar.markdown('---')
st.sidebar.info('Project Status: Model ready; UI in progress. Next: Deployment (Phase 3).')