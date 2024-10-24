import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the industry data from an Excel file
@st.cache_data
def load_industry_data():
    return pd.read_excel("IIP2024.xlsx")

# Function to load user data
@st.cache_data
def load_user_data(file):
    return pd.read_csv(file)

# Function to prepare data for modeling
def prepare_data(data, features, target):
    X = data[features]
    y = data[target]
    return X, y

# Function to plot actual vs predicted
def plot_predictions(y, lr_pred, rf_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Actual Data'))
    fig.add_trace(go.Scatter(x=y.index, y=lr_pred, mode='lines', name='Linear Regression Prediction'))
    fig.add_trace(go.Scatter(x=y.index, y=rf_pred, mode='lines', name='Random Forest Prediction'))

    fig.update_layout(
        title="Model Predictions vs Actual",
        xaxis_title="Index",
        yaxis_title="Target",
        hovermode="x unified"
    )
    return fig

# Streamlit app starts here
st.title("Industry and Financial Data Prediction")
st.sidebar.header("Upload Your Data")

# Step 1: Load the industry data
industry_data = load_industry_data()
st.write("Industry Data Preview:", industry_data.head())

# Step 2: Allow user to upload synthetic data
synthetic_data_file = st.sidebar.file_uploader("Upload Synthetic Data (CSV)", type=["csv"])
if synthetic_data_file is not None:
    user_synthetic_data = load_user_data(synthetic_data_file)
    st.write("User Synthetic Data Preview:", user_synthetic_data.head())

    # Step 3: Allow user to select features and target
    features = st.multiselect("Select Features", user_synthetic_data.columns.tolist(), default=user_synthetic_data.columns.tolist()[:-1])
    target = st.selectbox("Select Target Variable", user_synthetic_data.columns.tolist(), index=-1)

    if len(features) > 0 and target:
        # Prepare data
        X, y = prepare_data(user_synthetic_data, features, target)

        # Step 4: Train models
        st.subheader("Training Models")

        # Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        lr_pred = lr_model.predict(X)
        lr_rmse = mean_squared_error(y, lr_pred, squared=False)
        st.write(f"Linear Regression RMSE: {lr_rmse:.2f}")

        # Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)
        rf_rmse = mean_squared_error(y, rf_pred, squared=False)
        st.write(f"Random Forest RMSE: {rf_rmse:.2f}")

        # Step 5: Plot predictions
        fig = plot_predictions(y, lr_pred, rf_pred)
        st.plotly_chart(fig, use_container_width=True)

        # Step 6: Save trained models for download
        st.subheader("Download Trained Models")

        # Save Linear Regression Model
        lr_model_file = "linear_regression_model.pkl"
        with open(lr_model_file, 'wb') as f:
            pickle.dump(lr_model, f)
        st.download_button("Download Linear Regression Model", data=open(lr_model_file, 'rb'), file_name=lr_model_file)

        # Save Random Forest Model
        rf_model_file = "random_forest_model.pkl"
        with open(rf_model_file, 'wb') as f:
            pickle.dump(rf_model, f)
        st.download_button("Download Random Forest Model", data=open(rf_model_file, 'rb'), file_name=rf_model_file)

# Step 7: Optionally, allow the user to upload a pre-trained model for prediction
st.sidebar.header("Upload Pre-trained Model for Prediction")
model_file = st.sidebar.file_uploader("Upload Trained Model (Pickle)", type=["pkl"])
if model_file is not None and synthetic_data_file is not None:
    loaded_model = pickle.load(model_file)
    st.write("Pre-trained model loaded.")

    # Allow user to input values for predictions
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Input value for {feature}:", value=float(X[feature].iloc[-1]))

    input_df = pd.DataFrame([input_data])

    # Make prediction with the uploaded model
    prediction = loaded_model.predict(input_df)
    st.write(f"Prediction from uploaded model: {prediction[0]:.2f}")
