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

iip_data, synthetic_data, stock_data, correlation_results, financial_data = load_data()

# Define Industry and Indicators
indicators = {
    'Manufacture of Food Products': {
        'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
        'Lagging': ['Inventory Levels', 'Employment Data']
    },
    'Manufacture of Beverages': {
        'Leading': ['Consumer Confidence', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Profit Margins']
    },
    'Manufacture of Tobacco Products': {
        'Leading': ['Regulatory Changes', 'Consumer Trends'],
        'Lagging': ['Sales Volume', 'Market Share']
    },
    'Manufacture of Textiles': {
        'Leading': ['Fashion Trends', 'Raw Material Prices'],
        'Lagging': ['Export Data', 'Inventory Levels']
    },
    'Manufacture of Wearing Apparel': {
        'Leading': ['Retail Sales of Apparel', 'Consumer Spending on Fashion'],
        'Lagging': ['Production and Sales Data', 'Employment Trends in Apparel Sector']
    },
    'Manufacture of Leather and Related Products': {
        'Leading': ['Fashion Industry Trends', 'Raw Material Prices (Leather)'],
        'Lagging': ['Sales and Revenue Data', 'Inventory Levels']
    },
    'Manufacture of Wood and Products of Wood and Cork': {
        'Leading': ['Housing Market Data', 'Building Permits'],
        'Lagging': ['Production Volume', 'Employment Data']
    },
    'Manufacture of Paper and Paper Products': {
        'Leading': ['Consumer Spending on Paper Goods', 'Raw Material Prices (Wood Pulp)'],
        'Lagging': ['Production Output', 'Sales Data']
    },
    'Printing and Reproduction of Recorded Media': {
        'Leading': ['Trends in Media Consumption', 'Technological Advances'],
        'Lagging': ['Production Volume', 'Revenue and Profit Margins']
    },
    'Manufacture of Coke and Refined Petroleum Products': {
        'Leading': ['Crude Oil Prices', 'Energy Demand Trends'],
        'Lagging': ['Refined Product Output', 'Profit Margins']
    },
    'Manufacture of Chemicals and Chemical Products': {
        'Leading': ['Raw Material Prices', 'Industrial Production Data'],
        'Lagging': ['Production Data', 'Sales Revenue']
    },
    'Manufacture of Pharmaceuticals, Medicinal Chemicals, and Botanical Products': {
        'Leading': ['Regulatory Approvals', 'Research and Development Investment'],
        'Lagging': ['Drug Sales Data', 'Profit Margins']
    },
    'Manufacture of Rubber and Plastics Products': {
        'Leading': ['Raw Material Prices', 'Automotive Industry Trends'],
        'Lagging': ['Production and Sales Data', 'Inventory Levels']
    },
    'Manufacture of Other Non-Metallic Mineral Products': {
        'Leading': ['Construction and Infrastructure Projects', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Sales Revenue']
    },
    'Manufacture of Basic Metals': {
        'Leading': ['Industrial Production', 'Raw Material Prices'],
        'Lagging': ['Production Data', 'Employment Trends']
    },
    'Manufacture of Fabricated Metal Products, Except Machinery and Equipment': {
        'Leading': ['Construction and Manufacturing Activity', 'Raw Material Prices'],
        'Lagging': ['Production Volume', 'Sales Data']
    },
    'Manufacture of Computer, Electronic, and Optical Products': {
        'Leading': ['Technology Trends', 'Consumer Electronics Sales'],
        'Lagging': ['Sales Revenue', 'Production Output']
    },
    'Manufacture of Electrical Equipment': {
        'Leading': ['Industrial Production Trends', 'Investment in Infrastructure'],
        'Lagging': ['Production and Sales Data', 'Profit Margins']
    },
    'Manufacture of Machinery and Equipment n.e.c.': {
        'Leading': ['Capital Expenditure', 'Industrial Production Data'],
        'Lagging': ['Production Data', 'Sales Revenue']
    },
    'Manufacture of Motor Vehicles, Trailers, and Semi-Trailers': {
        'Leading': ['Automobile Sales Trends', 'Consumer Confidence'],
        'Lagging': ['Production and Sales Data', 'Employment Trends']
    },
    'Manufacture of Other Transport Equipment': {
        'Leading': ['Transportation Infrastructure Investment', 'Global Trade Trends'],
        'Lagging': ['Production Output', 'Sales Revenue']
    },
    'Manufacture of Furniture': {
        'Leading': ['Housing Market Trends', 'Consumer Spending Trends'],
        'Lagging': ['Production and Sales Data', 'Inventory Levels']
    },
    'Other Manufacturing': {
        'Leading': ['Sector-Specific Trends', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Sales Data']
    },
    'India Inflation CPI (Consumer Price Index)': {
        'Leading': ['Consumer Spending Trends', 'Wage Growth', 'Raw Material Prices'],
        'Lagging': ['Previous CPI Data', 'Retail Sales Data', 'Employment Cost Index']
    },
    'India Inflation WPI (Wholesale Price Index)': {
        'Leading': ['Producer Price Trends', 'Raw Material Prices', 'Commodity Prices'],
        'Lagging': ['Previous WPI Data', 'Import and Export Prices', 'Manufacturing Output Data']
    },
    'India GDP Growth': {
        'Leading': ['Business Investment', 'Consumer Confidence Index', 'Industrial Production'],
        'Lagging': ['Previous GDP Data', 'Employment Data', 'Corporate Earnings']
    },
    'RBI Interest Rate': {
        'Leading': ['Inflation Data', 'Economic Growth Data', 'Global Interest Rates'],
        'Lagging': ['Previous RBI Interest Rates', 'Credit Growth Data', 'Inflation Adjustments']
    },
    'India Infrastructure Output': {
        'Leading': ['Building Permits', 'Government Infrastructure Spending', 'Construction Sector Activity'],
        'Lagging': ['Previous Infrastructure Output Data', 'Project Completion Rates', 'Employment in Construction Sector']
    },
    'India Banks Loan Growth Rate': {
        'Leading': ['Consumer Confidence', 'Business Investment Trends', 'Interest Rate Trends'],
        'Lagging': ['Previous Loan Growth Data', 'Bank Credit Demand', 'Non-Performing Loans']
    },
    'India Forex FX Reserves': {
        'Leading': ['Trade Balance Data', 'Foreign Investment Inflows', 'Global Economic Conditions'],
        'Lagging': ['Previous Forex Reserves Data', 'Currency Exchange Rates', 'International Financial Assistance']
    }
}

# Function to interpret correlation
def interpret_correlation(value):
    if value > 0.8:
        return "Strong Positive"
    elif 0.3 < value <= 0.8:
        return "Slight Positive"
    elif -0.3 <= value <= 0.3:
        return "Neutral"
    elif -0.8 <= value < -0.3:
        return "Slight Negative"
    else:
        return "Strong Negative"

# Function to adjust correlation values based on prediction
def adjust_correlation(correlation_value, predicted_value, industry_mean):
    return correlation_value * (predicted_value / industry_mean)

# Function to get detailed interpretation with Indian economy context
def get_detailed_interpretation(parameter_name, correlation_interpretation):
    interpretations = {
        "correlation with Total Revenue/Income": {
            "Slight Positive": (
            ),
            "Strong Positive": (
            ),
            "Neutral": (
            ),
            "Slight Negative": (
            ),
            "Strong Negative": (
            )
        },
        "correlation with Total Operating Expense": {
            "Slight Positive": (
            ),
            "Strong Positive": (
            ),
            "Neutral": (
            ),
            "Slight Negative": (
            ),
            "Strong Negative": (
            )
        },
    }
    return interpretations.get(parameter_name, {}).get(correlation_interpretation, "No interpretation available.")


# Function to get the latest financial data
def get_latest_financial_data(stock_name):
    if stock_name in financial_data:
        stock_financial_data = financial_data[stock_name]
        balance_sheet = stock_financial_data.get('BalanceSheet', pd.DataFrame())
        income_statement = stock_financial_data.get('IncomeStatement', pd.DataFrame())
        cash_flow = stock_financial_data.get('CashFlow', pd.DataFrame())
        
        latest_balance_sheet = balance_sheet[balance_sheet['Date'] == 'Dec 2023'].iloc[-1] if not balance_sheet.empty else pd.Series()
        latest_income_statement = income_statement[income_statement['Date'] == 'Jun 2024'].iloc[-1] if not income_statement.empty else pd.Series()
        latest_cash_flow = cash_flow[cash_flow['Date'] == 'Dec 2023'].iloc[-1] if not cash_flow.empty else pd.Series()
        
        return latest_balance_sheet, latest_income_statement, latest_cash_flow
    else:
        return pd.Series(), pd.Series(), pd.Series()

# Streamlit App Interface
st.title('Adhuniq Industry and Financial Data Prediction11')
st.sidebar.header('Select Options')

selected_industry = st.sidebar.selectbox(
    'Select Industry',
    list(indicators.keys()),  # Use the keys from the indicators dictionary
    index=0  # Default index if needed, here 'Manufacture of Food Products'
)

# New Feature: Input for CPI and Interest Rate
expected_cpi = st.sidebar.number_input('Expected CPI (%):', min_value=0.0, value=5.0)
expected_interest_rate = st.sidebar.number_input('Expected RBI Interest Rate (%):', min_value=0.0, value=6.0)

# Scenario Selection
scenario = st.sidebar.selectbox('Select Scenario', ['Base Case', 'Best Case', 'Worst Case'])

if selected_industry:
    # Normalize and match the industry name with sheet names
    normalized_industry = selected_industry.strip().lower()
    matched_sheet_name = None
    
    for sheet_name in synthetic_data.keys():
        if sheet_name.strip().lower() == normalized_industry:
            matched_sheet_name = sheet_name
            break
    
    if matched_sheet_name:
        st.header(f'Industry: {selected_industry}')
        
        if selected_industry in indicators:
            # Prepare Data for Modeling
            def prepare_data(industry, data, iip_data):
                leading_indicators = indicators[industry]['Leading']
                
                X = data[leading_indicators].shift(1).dropna()
                y = iip_data[industry].loc[X.index]
                return X, y
            
            X, y = prepare_data(selected_industry, synthetic_data[matched_sheet_name], iip_data)
            
            # Regression Model
            reg_model = LinearRegression()
            reg_model.fit(X, y)
            reg_pred = reg_model.predict(X)

            # ARIMA Model
            arima_model = ARIMA(y, order=(5, 1, 0))  # Adjust order parameters as needed
            arima_result = arima_model.fit()
            arima_pred = arima_result.predict(start=1, end=len(y), dynamic=False)

            # Machine Learning Model (Random Forest)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)

            # Model Comparison
            st.subheader('Model Performance')
            st.write(f"Linear Regression RMSE: {mean_squared_error(y, reg_pred, squared=False):.2f}")
            st.write(f"ARIMA RMSE: {mean_squared_error(y, arima_pred, squared=False):.2f}")
            st.write(f"Random Forest RMSE: {mean_squared_error(y, rf_pred, squared=False):.2f}")

            # Visualization
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
            for indicator in indicators[selected_industry]['Leading']:
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
            latest_leading_indicators_data = synthetic_data[matched_sheet_name][indicators[selected_industry]['Leading']].tail()
            st.write('**Leading Indicators Data:**')
            st.write(latest_leading_indicators_data)

            # Lagging Indicators Data
            latest_lagging_indicators_data = synthetic_data[matched_sheet_name][indicators[selected_industry]['Lagging']].tail()
            st.write('**Lagging Indicators Data:**')
            st.write(latest_lagging_indicators_data)

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
                                    updated_corr_data[f'Interpreted {col}'] = updated_corr_data[col].apply(interpret_correlation)
                                    updated_corr_data[f'Adjusted {col}'] = updated_corr_data.apply(
                                        lambda row: adjust_correlation(row[col], row['Predicted Industry Value'], industry_mean),
                                        axis=1
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

                        # Preparing data for plotting
                        actual_corr_cols = [col for col in correlation_results.columns if 'correlation' in col]
                        predicted_corr_cols = [f'Adjusted {col}' for col in actual_corr_cols if f'Adjusted {col}' in combined_corr_data.columns]

                        fig = go.Figure()
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

                        # Calculate Predicted Income Statement Results
                        st.subheader('Predicted Income Statement Results')

                        # Initialize a DataFrame to store predicted results for each stock
                        all_predicted_results = []

                        for stock in selected_stocks:
                            income_statement_data = financial_data.get(stock, {}).get('IncomeStatement', pd.DataFrame())
                            income_statement_data = income_statement_data[income_statement_data['Date'] == 'Jun 2024'].iloc[-1] if not income_statement_data.empty else pd.Series()
                            
                            if not income_statement_data.empty:
                                income_statement_values = income_statement_data.dropna()
                                income_statement_dict = income_statement_values.to_dict()
                                
                                # Create a DataFrame for correlation and income statement data
                                corr_cols = [
                                    'Adjusted correlation with Total Revenue/Income',
                                    'Adjusted correlation with Net Income',
                                    'Adjusted correlation with Total Operating Expense',
                                    'Adjusted correlation with Operating Income/Profit',
                                    'Adjusted correlation with EBITDA',
                                    'Adjusted correlation with EBIT',
                                    'Adjusted correlation with Income/Profit Before Tax',
                                    'Adjusted correlation with Net Income From Continuing Operation',
                                    'Adjusted correlation with Net Income',
                                    'Adjusted correlation with Net Income Applicable to Common Share',
                                    'Adjusted correlation with EPS (Earning Per Share)'
                                ]
                                
                                # Initialize a dictionary to store the predicted results
                                predicted_results = {}
                                
                                for col in corr_cols:
                                    if col in combined_corr_data.columns:
                                        correlation_value = combined_corr_data[combined_corr_data['Stock Name'] == stock].iloc[0].get(col, 0)
                                        statement_value = income_statement_dict.get(col.replace('Adjusted correlation with ', ''), 0)
                                        predicted_results[col.replace('Adjusted correlation with ', '')] = (statement_value * correlation_value) + statement_value
                                
                                # Convert the predicted results dictionary to a DataFrame
                                predicted_results_df = pd.DataFrame(predicted_results, index=[f'Predicted Income Statement Result ({stock})']).T
                                all_predicted_results.append(predicted_results_df)
                                
                                # Display Latest Financial Data
                                st.write(f"**Latest Financial Data for {stock}:**")
                                if not income_statement_data.empty:
                                    st.write("**Income Statement (Jun 2024):**")
                                    st.write(income_statement_data)
                                  
                                # Display the full predicted results
                                st.write(f"**Predicted Income Statement Results for {stock}:**")
                                st.write(predicted_results_df)
                        
                        # Combine all predicted results
                        if all_predicted_results:
                            combined_predicted_results = pd.concat(all_predicted_results)
                            st.write('**All Predicted Income Statement Results:**')
                            st.write(combined_predicted_results)

                            # Display Detailed Interpretation with Indian Economy Context
                            st.write('**Detailed Interpretation with Indian Economy Context:**')
                            
                            parameters = [
                                'Total Revenue/Income',
                                'Total Operating Expense',
                                'Operating Income/Profit',
                                'EBITDA',
                                'EBIT',
                                'Income/Profit Before Tax',
                                'Net Income From Continuing Operation',
                                'Net Income',
                                'Net Income Applicable to Common Share',
                                'EPS (Earning Per Share)'
                            ]

                            for parameter in parameters:
                                for stock in selected_stocks:
                                    correlation_value = combined_corr_data[
                                        combined_corr_data['Stock Name'] == stock
                                    ].iloc[0].get(f'Interpreted correlation with {parameter}', "Neutral")
                                    
                                    interpretation = get_detailed_interpretation(f'correlation with {parameter}', correlation_value)
                                    st.write(f"**{stock} - correlation with {parameter}:**")
                                    st.write(interpretation)

else:
    st.error('No correlation results available.')

