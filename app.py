import os
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

# Load Data
@st.cache_data
def load_data():
    # Load basic industrial data
    iip_data = pd.read_excel('IIP2024.xlsx')
    synthetic_data = pd.read_excel('Synthetic_Industry_Data.xlsx', sheet_name=None)
    
    return iip_data, synthetic_data

# Load data
iip_data, synthetic_data = load_data()

# Define Industry and Indicators (without stock data)
indicators = {
    'Manufacture of Food Products': {
        'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
        'Lagging': ['Inventory Levels', 'Employment Data']
    },
    'Manufacture of Beverages': {
        'Leading': ['Consumer Confidence', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Profit Margins']
    },
    'Manufacture of Textiles': {
        'Leading': ['Fashion Trends', 'Raw Material Prices'],
        'Lagging': ['Export Data', 'Inventory Levels']
    },
}

# Streamlit app interface
st.title('Industry Indicator Prediction App')

# Select Industry
industry = st.selectbox('Select Industry:', list(indicators.keys()))

# Display selected industry leading and lagging indicators
st.header(f'{industry} Indicators')
st.subheader('Leading Indicators')
for indicator in indicators[industry]['Leading']:
    st.write(f'- {indicator}')
    
st.subheader('Lagging Indicators')
for indicator in indicators[industry]['Lagging']:
    st.write(f'- {indicator}')

# User inputs for prediction
expected_cpi = st.number_input('Expected CPI (%):', value=5.00, step=0.01)
expected_interest_rate = st.number_input('Expected RBI Interest Rate (%):', value=6.00, step=0.01)
st.subheader('Predict Future Values')

# Input fields for leading indicators
expected_consumer_spending = st.number_input('Expected Consumer Spending Trends Value:', value=64.61, step=0.01)
expected_agricultural_output = st.number_input('Expected Agricultural Output Value:', value=32.47, step=0.01)
expected_retail_sales = st.number_input('Expected Retail Sales Data Value:', value=75.50, step=0.01)

# Prediction Logic (a simple weighted average as an example)
base_prediction = (expected_consumer_spending + expected_agricultural_output + expected_retail_sales) / 3
adjusted_prediction = base_prediction * (1 + (expected_cpi - 5.00) / 100) * (1 - (expected_interest_rate - 6.00) / 100)

# Display the prediction
st.subheader('Adjusted Prediction:')
st.write(f'Base Case: {base_prediction:.2f}')
st.write(f'Adjusted for CPI and Interest Rate: {adjusted_prediction:.2f}')

# Create and display a simple graph of the predicted values
fig = go.Figure()
fig.add_trace(go.Scatter(x=['Base Case', 'Adjusted Prediction'], y=[base_prediction, adjusted_prediction], mode='lines+markers'))

# Set graph titles
fig.update_layout(title=f'Prediction for {industry}', xaxis_title='Prediction Type', yaxis_title='Predicted Value')
st.plotly_chart(fig)
