import os
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    iip_data = pd.read_excel('IIP2024.xlsx')
    synthetic_data = pd.read_excel('Synthetic_Industry_Data.xlsx', sheet_name=None)
    
    stock_data = {}
    stock_data_folder = 'stockdata'
    for filename in os.listdir(stock_data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv', '')
            stock_data[stock_name] = pd.read_csv(os.path.join(stock_data_folder, filename))
    
    # Load correlation results
    correlation_results = pd.read_excel(os.path.join(stock_data_folder, 'Manufacture_of_Food_Products_correlation_results.xlsx'))
    
    # Load financial data
    financial_data = {}
    financial_folder = 'financial'
    for filename in os.listdir(financial_folder):
        if filename.endswith('.xlsx'):
            stock_name = filename.replace('.xlsx', '')
            stock_file_path = os.path.join(financial_folder, filename)
            financial_data[stock_name] = pd.read_excel(stock_file_path, sheet_name=None)
    
    return iip_data, synthetic_data, stock_data, correlation_results, financial_data

# Function to load user-uploaded data
@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file:
        return pd.read_excel(uploaded_file, sheet_name=None)
    return None

iip_data, synthetic_data, stock_data, correlation_results, financial_data = load_data()

# Streamlit App Interface
st.title('Adhuniq Industry and Financial Data Prediction')

# Upload Leading Indicators Data
uploaded_file = st.sidebar.file_uploader("Upload Leading Indicators Data (Excel)", type=["xlsx"])

if uploaded_file:
    user_uploaded_data = load_uploaded_data(uploaded_file)
    st.write("User Uploaded Data:")
    st.write(user_uploaded_data)

selected_industry = st.sidebar.selectbox(
    'Select Industry',
    list(synthetic_data.keys()),  # Use the keys from the existing synthetic data
    index=0  # Default index if needed
)

# New Feature: Input for CPI and Interest Rate
expected_cpi = st.sidebar.number_input('Expected CPI (%):', min_value=0.0, value=5.0)
expected_interest_rate = st.sidebar.number_input('Expected RBI Interest Rate (%):', min_value=0.0, value=6.0)

# Scenario Selection
scenario = st.sidebar.selectbox('Select Scenario', ['Base Case', 'Best Case', 'Worst Case'])

# Allow predictions with uploaded data
if uploaded_file and selected_industry:
    st.header(f'Industry: {selected_industry}')

    # Prepare Data for Modeling
    def prepare_data(industry, data, iip_data):
        leading_indicators = list(data.columns)  # Use the columns from the uploaded data

        X = data[leading_indicators].shift(1).dropna()
        y = iip_data[industry].loc[X.index]
        return X, y
    
    # Use the user-uploaded data for modeling
    X, y = prepare_data(selected_industry, user_uploaded_data[selected_industry], iip_data)
    
    # Train models
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    reg_pred = reg_model.predict(X)

    arima_model = ARIMA(y, order=(5, 1, 0))  # Adjust order parameters as needed
    arima_result = arima_model.fit()
    arima_pred = arima_result.predict(start=1, end=len(y), dynamic=False)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict(X)

    # Display Model Performance
    st.subheader('Model Performance with Uploaded Data')
    st.write(f"Linear Regression RMSE: {mean_squared_error(y, reg_pred, squared=False):.2f}")
    st.write(f"ARIMA RMSE: {mean_squared_error(y, arima_pred, squared=False):.2f}")
    st.write(f"Random Forest RMSE: {mean_squared_error(y, rf_pred, squared=False):.2f}")

    # Visualization of Predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Actual Industry Data'))
    fig.add_trace(go.Scatter(x=y.index, y=reg_pred, mode='lines', name='Linear Regression Prediction'))
    fig.add_trace(go.Scatter(x=y.index, y=arima_pred, mode='lines', name='ARIMA Prediction'))
    fig.add_trace(go.Scatter(x=y.index, y=rf_pred, mode='lines', name='Random Forest Prediction'))

    fig.update_layout(
        title=f'Industry Data Prediction for {selected_industry}',
        xaxis_title='Date',
        yaxis_title='Industry Data',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Predict Future Values
    st.subheader('Predict Future Values')

    input_data = {}
    for indicator in user_uploaded_data[selected_industry].columns:
        input_data[indicator] = st.number_input(f'Expected {indicator} Value:', value=float(X[indicator].iloc[-1]))

    input_df = pd.DataFrame(input_data, index=[0])

    # Predictions
    future_reg_pred = reg_model.predict(input_df)
    future_rf_pred = rf_model.predict(input_df)

    # Adjust predictions based on selected scenario
    adjustment_factors = {
        'Base Case': 1.0,
        'Best Case': 1.1,  # 10% increase for best case
        'Worst Case': 0.9,  # 10% decrease for worst case
    }

    adjusted_reg_pred = future_reg_pred * adjustment_factors[scenario]
    adjusted_rf_pred = future_rf_pred * adjustment_factors[scenario]

    # Impact Analysis based on CPI and Interest Rate
    adjusted_industry_value = adjusted_reg_pred[0] * (1 - (expected_cpi / 100)) * (1 - (expected_interest_rate / 100))
    st.write(f"Adjusted Prediction considering CPI and Interest Rate ({scenario}): {adjusted_industry_value:.2f}")

    # Display Latest Data
    st.subheader('Adhuniq Industry and Indicator Data')

    # Industry Data
    latest_industry_data = iip_data[[selected_industry]].tail()  # Show the last few rows
    st.write('**Industry Data:**')
    st.write(latest_industry_data)

    # Leading Indicators Data
    latest_leading_indicators_data = user_uploaded_data[selected_industry].tail()
    st.write('**Leading Indicators Data:**')
    st.write(latest_leading_indicators_data)

# Stock Selection and Correlation Analysis
if correlation_results is not None:
    # Allow multiple stock selection
    selected_stocks = st.sidebar.multiselect('Select Stocks', correlation_results['Stock Name'].tolist())

    if selected_stocks:
        # Filter correlation results for selected stocks
        selected_corr_data = correlation_results[correlation_results['Stock Name'].isin(selected_stocks)]
        
        # Initialize a DataFrame to store adjusted correlations
        all_adjusted_corr_data = []

        for stock in selected_stocks:
            st.subheader(f'Correlation Analysis with {stock}')
            
            # Fetch correlation data for the selected stock
            stock_correlation_data = selected_corr_data[selected_corr_data['Stock Name'] == stock]

            if not stock_correlation_data.empty:
                st.write('**Actual Correlation Results:**')
                st.write(stock_correlation_data)

                # Prepare predicted correlation data
                st.subheader('Predicted Correlation Analysis')

                industry_mean = y.mean()
                updated_corr_data = stock_correlation_data.copy()
                updated_corr_data['Predicted Industry Value'] = adjusted_industry_value

                for col in [
                    'correlation with Total Revenue/Income',
                    'correlation with Net Income',
                    'correlation with Total Operating Expense',
                    'correlation with Operating Income/Profit',
                    'correlation with EBITDA',
                    'correlation with EBIT',
                    'correlation with Income/Profit Before Tax',
                    'correlation with Net Income From Continuing Operation',
                    'correlation with Net Income Applicable to Common Share',
                    'correlation with EPS (Earning Per Share)',
                    'correlation with Operating Margin',
                    'correlation with EBITDA Margin',
                    'correlation with Net Profit Margin',
                    'Annualized correlation with Total Revenue/Income',
                    'Annualized correlation with Total Operating Expense',
                    'Annualized correlation with Operating Income/Profit',
                    'Annualized correlation with EBITDA',
                    'Annualized correlation with EBIT',
                    'Annualized correlation with Income/Profit Before Tax',
                    'Annualized correlation with Net Income From Continuing Operation',
                    'Annualized correlation with Net Income',
                    'Annualized correlation with Net Income Applicable to Common Share',
                    'Annualized correlation with EPS (Earning Per Share)'
                ]:
                    if col in updated_corr_data.columns:
                        updated_corr_data[f'Adjusted {col}'] = updated_corr_data.apply(
                            lambda row: row[col] * (row['Predicted Industry Value'] / industry_mean), axis=1
                        )
                
                all_adjusted_corr_data.append(updated_corr_data)
            
        # Combine all adjusted correlation data
        if all_adjusted_corr_data:
            combined_corr_data = pd.concat(all_adjusted_corr_data, ignore_index=True)

            st.write('**Predicted Correlation Results:**')
            st.write(combined_corr_data[['Stock Name', 'Predicted Industry Value'] +
                                        [col for col in combined_corr_data.columns if 'Adjusted' in col]])

            # Interactive Comparison Chart
            st.subheader('Interactive Comparison of Actual and Predicted Correlation Results')

            fig = go.Figure()
            actual_corr_cols = [col for col in correlation_results.columns if 'correlation' in col]
            predicted_corr_cols = [f'Adjusted {col}' for col in actual_corr_cols if f'Adjusted {col}' in combined_corr_data.columns]

            for col in actual_corr_cols:
                if col in selected_corr_data.columns:
                    fig.add_trace(go.Bar(
                        x=selected_corr_data['Stock Name'],
                        y=selected_corr_data[col],
                        name=f'Actual {col}',
                        marker_color='blue'
                    ))

            for col in predicted_corr_cols:
                if col in combined_corr_data.columns:
                    fig.add_trace(go.Bar(
                        x=combined_corr_data['Stock Name'],
                        y=combined_corr_data[col],
                        name=f'Predicted {col}',
                        marker_color='orange'
                    ))

            fig.update_layout(
                title='Comparison of Actual and Predicted Correlation Results',
                xaxis_title='Stock',
                yaxis_title='Correlation Value',
                barmode='group',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
else:
    st.error('No correlation results available.')
