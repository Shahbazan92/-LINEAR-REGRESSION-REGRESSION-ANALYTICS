import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Define the custom FeatureEngineer class consistent with training
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["TotalArea"] = X["GrLivArea"] + X["TotalBsmtSF"]
        X["TotalBathrooms"] = (
            X["FullBath"]
            + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"]
            + 0.5 * X["BsmtHalfBath"]
        )
        return X

st.set_page_config(page_title="Ames Housing Price Predictor", layout="wide")

# 2. Load the complete saved model package containing metadata defaults
@st.cache_resource
def load_model_package():
    return joblib.load("house_price_model.pkl")

try:
    package = load_model_package()
    model = package["model"]
    feature_columns = package["feature_columns"]
    numeric_defaults = package["numeric_defaults"]
    categorical_defaults = package["categorical_defaults"]
    category_options = package["category_options"]
    st.success("🚀 Machine Learning pipeline and training metadata loaded successfully!")
except Exception as e:
    st.error(f"Error loading model components: {e}")
    st.stop()

st.title("🏡 Ames, Iowa Housing Price Predictor")
st.write("This interactive app uses your optimized model to predict house sales prices in Ames, Iowa.")

st.sidebar.header("🔧 Input Property Details")

# Prepare input dictionary populated with the official training defaults
input_data = {}
for col in feature_columns:
    if col in numeric_defaults:
        input_data[col] = numeric_defaults[col]
    elif col in categorical_defaults:
        input_data[col] = categorical_defaults[col]

with st.sidebar:
    st.subheader("📐 Dimensions & Space")
    if "GrLivArea" in feature_columns:
        input_data["GrLivArea"] = st.number_input("GrLivArea (Above ground living area SqFt)", min_value=100, max_value=10000, value=int(numeric_defaults.get("GrLivArea", 1400)))
    if "TotalBsmtSF" in feature_columns:
        input_data["TotalBsmtSF"] = st.number_input("TotalBsmtSF (Total Basement Area SqFt)", min_value=0, max_value=10000, value=int(numeric_defaults.get("TotalBsmtSF", 1000)))
    if "LotArea" in feature_columns:
        input_data["LotArea"] = st.number_input("LotArea (Lot size in SqFt)", min_value=100, max_value=100000, value=int(numeric_defaults.get("LotArea", 9000)))

    st.subheader("⭐ Quality & Timeline")
    if "OverallQual" in feature_columns:
        input_data["OverallQual"] = st.slider("OverallQual (Material Quality 1-10)", 1, 10, int(numeric_defaults.get("OverallQual", 6)))
    if "OverallCond" in feature_columns:
        input_data["OverallCond"] = st.slider("OverallCond (Condition Rating 1-10)", 1, 10, int(numeric_defaults.get("OverallCond", 5)))
    if "YearBuilt" in feature_columns:
        input_data["YearBuilt"] = st.number_input("YearBuilt (Original Construction Year)", min_value=1800, max_value=2026, value=int(numeric_defaults.get("YearBuilt", 1970)))
    if "YearRemodAdd" in feature_columns:
        input_data["YearRemodAdd"] = st.number_input("YearRemodAdd (Remodel Year)", min_value=1950, max_value=2026, value=int(numeric_defaults.get("YearRemodAdd", 1990)))
    if "YrSold" in feature_columns:
        input_data["YrSold"] = st.number_input("YrSold (Year Sold)", min_value=2006, max_value=2026, value=int(numeric_defaults.get("YrSold", 2008)))

    st.subheader("🛁 Rooms & Garage")
    if "FullBath" in feature_columns:
        input_data["FullBath"] = st.number_input("FullBath (Full Bathrooms above grade)", min_value=0, max_value=10, value=int(numeric_defaults.get("FullBath", 2)))
    if "HalfBath" in feature_columns:
        input_data["HalfBath"] = st.number_input("HalfBath (Half Bathrooms above grade)", min_value=0, max_value=5, value=int(numeric_defaults.get("HalfBath", 0)))
    if "BsmtFullBath" in feature_columns:
        input_data["BsmtFullBath"] = st.number_input("BsmtFullBath (Basement Full Bathrooms)", min_value=0, max_value=5, value=int(numeric_defaults.get("BsmtFullBath", 0)))
    if "BsmtHalfBath" in feature_columns:
        input_data["BsmtHalfBath"] = st.number_input("BsmtHalfBath (Basement Half Bathrooms)", min_value=0, max_value=5, value=int(numeric_defaults.get("BsmtHalfBath", 0)))
    if "GarageCars" in feature_columns:
        input_data["GarageCars"] = st.number_input("GarageCars (Garage Car Capacity)", min_value=0, max_value=10, value=int(numeric_defaults.get("GarageCars", 2)))

    st.subheader("📍 Location")
    if "Neighborhood" in feature_columns:
        options = category_options["Neighborhood"]
        default_val = categorical_defaults["Neighborhood"]
        default_idx = options.index(default_val) if default_val in options else 0
        input_data["Neighborhood"] = st.selectbox("Neighborhood Location within Ames", options=options, index=default_idx)

# Construct input DataFrame with exactly the features and order used during training
input_df = pd.DataFrame([input_data], columns=feature_columns)

st.header("📈 House Valuation")
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Below are the primary parameters passed to your pipeline:")
    st.dataframe(input_df[['GrLivArea', 'TotalBsmtSF', 'LotArea', 'OverallQual', 'YearBuilt', 'FullBath', 'GarageCars', 'Neighborhood']])

with col2:
    if st.button("Calculate Estimated Price", type="primary"):
        with st.spinner("Calculating estimated house price..."):
            try:
                # Generate predictions using the full DataFrame in training configuration
                pred_log_price = model.predict(input_df)[0]
                pred_dollar_price = np.expm1(pred_log_price)
                
                st.success("Prediction completed successfully!")
                st.metric(
                    label="Estimated Sale Price",
                    value=f"${pred_dollar_price:,.2f}"
                )
                st.balloons()
            except Exception as e:
                st.error(f"Prediction Error: {e}")
