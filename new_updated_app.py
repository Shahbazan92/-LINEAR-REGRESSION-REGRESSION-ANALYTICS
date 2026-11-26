import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Ames Housing Price Predictor", layout="wide")

# 1. Define the custom FeatureEngineer class so joblib can deserialize it
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

# 2. Load the complete saved end-to-end pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("house_price_complete_pipeline.pkl")

try:
    pipeline = load_pipeline()
    st.success("🚀 Complete end-to-end prediction pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()

st.title("🏡 Ames, Iowa Housing Price Predictor")
st.write("This interactive app uses an optimized Lasso Regression Pipeline to predict house sales prices in Ames, Iowa.")

# 3. Setup defaults for sidebar input columns
numeric_defaults = {
    'LotFrontage': 70.0, 'LotArea': 9600.0, 'YearBuilt': 1972, 'YearRemodAdd': 1994, 'MasVnrArea': 0.0,
    'BsmtFinSF1': 384.0, 'BsmtFinSF2': 0.0, 'BsmtUnfSF': 480.0, 'TotalBsmtSF': 997.0, '1stFlrSF': 1095.0,
    '2ndFlrSF': 0.0, 'LowQualFinSF': 0.0, 'GrLivArea': 1473.0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0,
    'FullBath': 2, 'HalfBath': 0, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'TotRmsAbvGrd': 6,
    'Fireplaces': 0, 'GarageYrBlt': 1980, 'GarageCars': 2, 'GarageArea': 480.0, 'WoodDeckSF': 0.0,
    'OpenPorchSF': 25.0, 'EnclosedPorch': 0.0, '3SsnPorch': 0.0, 'ScreenPorch': 0.0, 'PoolArea': 0.0,
    'MiscVal': 0.0, 'MoSold': 6, 'YrSold': 2008
}

neighborhood_options = [
    'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer',
    'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'IDOTRR', 'Timber', 'NoRidge',
    'StoneBr', 'SWISU', 'ClearCr', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blmngtn', 'Blueste'
]

st.sidebar.header("🔧 Input Property Details")

with st.sidebar:
    st.subheader("📐 Dimensions & Space")
    gr_liv_area = st.number_input("GrLivArea (Above ground living area SqFt)", min_value=300, max_value=6000, value=int(numeric_defaults['GrLivArea']))
    total_bsmt_sf = st.number_input("TotalBsmtSF (Total Basement Area SqFt)", min_value=0, max_value=6000, value=int(numeric_defaults['TotalBsmtSF']))
    lot_area = st.number_input("LotArea (Lot size in SqFt)", min_value=500, max_value=200000, value=int(numeric_defaults['LotArea']))

    st.subheader("⭐ Quality & Timeline")
    overall_qual = st.slider("OverallQual (Material and Finish Quality Rating 1-10)", 1, 10, 6)
    overall_cond = st.slider("OverallCond (Condition Rating 1-10)", 1, 10, 5)
    year_built = st.number_input("YearBuilt (Original Construction Year)", min_value=1800, max_value=2010, value=int(numeric_defaults['YearBuilt']))
    year_remod = st.number_input("YearRemodAdd (Remodel/Addition Year)", min_value=1950, max_value=2010, value=int(numeric_defaults['YearRemodAdd']))
    yr_sold = st.number_input("YrSold (Year Sold)", min_value=2006, max_value=2010, value=int(numeric_defaults['YrSold']))

    st.subheader("🛁 Rooms & Garage")
    full_bath = st.number_input("FullBath (Full Bathrooms above grade)", min_value=0, max_value=5, value=int(numeric_defaults['FullBath']))
    half_bath = st.number_input("HalfBath (Half Bathrooms above grade)", min_value=0, max_value=3, value=int(numeric_defaults['HalfBath']))
    bsmt_full_bath = st.number_input("BsmtFullBath (Basement Full Bathrooms)", min_value=0, max_value=3, value=int(numeric_defaults['BsmtFullBath']))
    bsmt_half_bath = st.number_input("BsmtHalfBath (Basement Half Bathrooms)", min_value=0, max_value=2, value=int(numeric_defaults['BsmtHalfBath']))
    garage_cars = st.number_input("GarageCars (Garage Car Capacity)", min_value=0, max_value=5, value=int(numeric_defaults['GarageCars']))

    st.subheader("📍 Location & Utilities")
    neighborhood = st.selectbox("Neighborhood Location within Ames", options=neighborhood_options)

# 4. Create full Pandas input row including the rest of the columns as default values
input_data = {}
for col, val in numeric_defaults.items():
    input_data[col] = [val]

categorical_defaults = {
    'MSZoning': 'RL', 'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl',
    'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Condition1': 'Norm', 'Condition2': 'Norm',
    'BldgType': '1Fam', 'HouseStyle': '1Story', 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd', 'MasVnrType': 'None', 'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc',
    'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'Unf', 'BsmtFinType2': 'Unf',
    'Heating': 'GasA', 'HeatingQC': 'TA', 'CentralAir': 'Y', 'Electrical': 'SBrkr', 'KitchenQual': 'TA',
    'Functional': 'Typ', 'FireplaceQu': 'None', 'GarageType': 'Attchd', 'GarageFinish': 'Unf', 'GarageQual': 'TA',
    'GarageCond': 'TA', 'PavedDrive': 'Y', 'PoolQC': 'None', 'Fence': 'None', 'MiscFeature': 'None',
    'SaleType': 'WD', 'SaleCondition': 'Normal'
}

for col, val in categorical_defaults.items():
    input_data[col] = [val]

input_data['GrLivArea'] = [gr_liv_area]
input_data['TotalBsmtSF'] = [total_bsmt_sf]
input_data['LotArea'] = [lot_area]
input_data['OverallQual'] = [overall_qual]
input_data['OverallCond'] = [overall_cond]
input_data['YearBuilt'] = [year_built]
input_data['YearRemodAdd'] = [year_remod]
input_data['YrSold'] = [yr_sold]
input_data['FullBath'] = [full_bath]
input_data['HalfBath'] = [half_bath]
input_data['BsmtFullBath'] = [bsmt_full_bath]
input_data['BsmtHalfBath'] = [bsmt_half_bath]
input_data['GarageCars'] = [garage_cars]
input_data['Neighborhood'] = [neighborhood]

input_df = pd.DataFrame(input_data)

st.header("📈 House Valuation")
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Below is the complete set of parameters passed to the preprocessing pipeline:")
    st.dataframe(input_df[['GrLivArea', 'TotalBsmtSF', 'LotArea', 'OverallQual', 'YearBuilt', 'FullBath', 'GarageCars', 'Neighborhood']])

with col2:
    if st.button("Calculate Estimated Price", type="primary"):
    
        with st.spinner("Calculating..."):

# Create DataFrame from Streamlit form inputs
input_df = pd.DataFrame([input_data])

# Add missing training features using training defaults
for col in feature_columns:

    if col not in input_df.columns:

        if pd.api.types.is_numeric_dtype(X_reference[col]):
            input_df[col] = X_reference[col].median()

        else:
            input_df[col] = X_reference[col].mode()[0]

# Keep exactly the same columns and order as training
input_df = input_df[feature_columns]

# Generate prediction
pred_log_price = pipeline.predict(input_df)[0]

# Convert log prediction back to original price
predicted_price = np.expm1(pred_log_price)

st.success(
    f"Estimated House Price: ${predicted_price:,.2f}"
)


            pred_log_price = pipeline.predict(input_df)[0]
            pred_dollar_price = np.expm1(pred_log_price)
            st.metric(
                label="Estimated Sale Price",
                value=f"${pred_dollar_price:,.2f}"
            )
            st.balloons()
