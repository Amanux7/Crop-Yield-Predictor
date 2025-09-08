# Updated utils.py with fix: Save expected_columns.pkl in load_and_preprocess_data instead of train_model, since features are available there.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import os
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Step 2/3: Create dictionary here (top-level, global)
# EXPANDED: Added more countries based on common Kaggle yield datasets (e.g., Albania to Zimbabwe)
# Sources: FAO STAT 2022, World Bank Arable Land Indicators (hectares)
country_area_ha = {
    'Albania': 691000,
    'Algeria': 8516000,
    'Angola': 5775000,
    'Argentina': 39759800,
    'Armenia': 523000,
    'Australia': 48769500,
    'Austria': 1336000,
    'Azerbaijan': 1921000,
    'Bahamas': 14000,
    'Bahrain': 8000,
    'Bangladesh': 7915000,
    'Barbados': 11000,
    'Belarus': 5663000,
    'Belgium': 1366000,
    'Belize': 150000,
    'Benin': 3450000,
    'Bhutan': 139000,
    'Bolivia': 4195000,
    'Bosnia and Herzegovina': 1026000,
    'Botswana': 249000,
    'Brazil': 80048500,
    'Bulgaria': 3290000,
    'Burkina Faso': 6500000,
    'Burundi': 1460000,
    'Cambodia': 4100000,
    'Cameroon': 7400000,
    'Canada': 51920500,
    'Cape Verde': 47000,
    'Central African Republic': 1900000,
    'Chad': 4900000,
    'Chile': 1848000,
    'China': 147692500,
    'Colombia': 1990000,
    'Comoros': 84000,
    'Congo': 530000,
    'Costa Rica': 235000,
    'Croatia': 1203000,
    'Cuba': 3032000,
    'Cyprus': 109000,
    'Czech Republic': 2456000,
    'Denmark': 2597000,
    'Djibouti': 1700,
    'Dominica': 8000,
    'Dominican Republic': 1050000,
    'Ecuador': 1443000,
    'Egypt': 3849000,
    'El Salvador': 761000,
    'Eritrea': 693000,
    'Estonia': 710000,
    'Eswatini': 176000,
    'Ethiopia': 15871000,
    'Fiji': 170000,
    'Finland': 2268000,
    'France': 18399000,
    'Gabon': 515000,
    'Gambia': 480000,
    'Georgia': 641000,
    'Germany': 11700000,
    'Ghana': 4700000,
    'Greece': 3660000,
    'Grenada': 5000,
    'Guatemala': 1850000,
    'Guinea': 3000000,
    'Guinea-Bissau': 300000,
    'Guyana': 490000,
    'Haiti': 1060000,
    'Honduras': 1480000,
    'Hungary': 4332000,
    'Iceland': 121000,
    'India': 176526000,
    'Indonesia': 47805500,
    'Iran': 16702000,
    'Iraq': 5230000,
    'Ireland': 1076000,
    'Israel': 379000,
    'Italy': 7108000,
    'Jamaica': 160000,
    'Japan': 4297000,
    'Jordan': 261000,
    'Kazakhstan': 24251600,
    'Kenya': 6110000,
    'Kiribati': 2000,
    'Kuwait': 8000,
    'Kyrgyzstan': 1286000,
    'Laos': 1500000,
    'Latvia': 1295000,
    'Lebanon': 136000,
    'Lesotho': 340000,
    'Liberia': 630000,
    'Lithuania': 2260000,
    'Luxembourg': 64000,
    'Madagascar': 3700000,
    'Malawi': 3800000,
    'Malaysia': 1880000,
    'Maldives': 3000,
    'Mali': 7150000,
    'Malta': 9000,
    'Marshall Islands': 2000,
    'Mauritania': 460000,
    'Mauritius': 37000,
    'Mexico': 25930100,
    'Micronesia': 3000,
    'Moldova': 1745000,
    'Mongolia': 1248000,
    'Montenegro': 18000,
    'Morocco': 9216000,
    'Mozambique': 6700000,
    'Myanmar': 13009000,
    'Namibia': 803000,
    'Nepal': 2340000,
    'Netherlands': 1049000,
    'New Zealand': 620000,
    'Nicaragua': 1560000,
    'Niger': 15900000,
    'Nigeria': 41293800,
    'North Korea': 2280000,
    'North Macedonia': 418000,
    'Norway': 815000,
    'Oman': 61000,
    'Pakistan': 22768600,
    'Palau': 1000,
    'Panama': 630000,
    'Papua New Guinea': 350000,
    'Paraguay': 4500000,
    'Peru': 4620000,
    'Philippines': 5710000,
    'Poland': 11102000,
    'Portugal': 1225000,
    'Qatar': 13000,
    'Romania': 8726000,
    'Russia': 126526700,
    'Rwanda': 1410000,
    'Saint Kitts and Nevis': 3000,
    'Saint Lucia': 5000,
    'Saint Vincent and the Grenadines': 3000,
    'Samoa': 8000,
    'Sao Tome and Principe': 5000,
    'Saudi Arabia': 3200000,
    'Senegal': 3390000,
    'Serbia': 3128000,
    'Seychelles': 1000,
    'Sierra Leone': 2300000,
    'Singapore': 1000,
    'Slovakia': 1366000,
    'Slovenia': 169000,
    'Solomon Islands': 23000,
    'Somalia': 1400000,
    'South Africa': 12500000,
    'South Korea': 1496000,
    'South Sudan': 2800000,
    'Spain': 12525000,
    'Sri Lanka': 1310000,
    'Sudan': 29628900,
    'Suriname': 67000,
    'Sweden': 2607000,
    'Switzerland': 1040000,
    'Syria': 4602000,
    'Taiwan': 810000,
    'Tajikistan': 860000,
    'Tanzania': 15000000,
    'Thailand': 16910000,
    'Timor-Leste': 160000,
    'Togo': 3700000,
    'Tonga': 6000,
    'Trinidad and Tobago': 22000,
    'Tunisia': 4657000,
    'Turkey': 24055300,
    'Turkmenistan': 1920000,
    'Tuvalu': 0,  # Minimal arable land
    'Uganda': 7200000,
    'Ukraine': 34767300,
    'United Arab Emirates': 42000,
    'United Kingdom': 6040000,
    'United States': 168182600,
    'Uruguay': 2040000,
    'Uzbekistan': 4300000,
    'Vanuatu': 15000,
    'Venezuela': 3100000,
    'Vietnam': 6828000,
    'Yemen': 1340000,
    'Zambia': 3850000,
    'Zimbabwe': 4100000,
    # Default for any missing: 10000000 (median approx)
}

def load_and_preprocess_data():
    # Load real data
    data = pd.read_csv('data/yield_df.csv')
    
    # Print columns to debug
    print("Original columns:", data.columns.tolist())
    
    # Rename columns to match PRD features
    data = data.rename(columns={
        'Area': 'Country',
        'Item': 'Crop_Type',
        'average_rain_fall_mm_per_year': 'Rainfall_mm',
        'avg_temp': 'Temperature_C',
        'pesticides_tonnes': 'Fertilizer_kg_ha',
        'hg/ha_yield': 'Yield_tons_ha'
    })
    data['Yield_tons_ha'] = data['Yield_tons_ha'] / 10000  # Correct conversion: hg/ha to tons/ha
    
    # Add placeholder Soil_pH
    data['Soil_pH'] = 6.5  # Average neutral pH
    
    # Preprocess: Encode, handle NaNs/outliers
    data = pd.get_dummies(data, columns=['Crop_Type'], drop_first=True)
    # Compute mean only on numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    # Filter realistic yield range
    data = data[(data['Yield_tons_ha'] > 0) & (data['Yield_tons_ha'] < 25)]
    
    # Feature engineering: Fit PolynomialFeatures with degree=1
    poly = PolynomialFeatures(degree=1, include_bias=False)
    numeric_features = ['Rainfall_mm', 'Fertilizer_kg_ha', 'Temperature_C', 'Soil_pH', 'Year']
    X_numeric = data[numeric_features]
    X_poly = poly.fit_transform(X_numeric)
    poly_feature_names = poly.get_feature_names_out(numeric_features)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=data.index)
    data = pd.concat([data.drop(columns=numeric_features), X_poly_df], axis=1)
    
    # Feature engineering: Interaction term
    data['Rain_Temp_Interaction'] = data['Rainfall_mm'] * data['Temperature_C']
    
    # Drop non-predictive or non-numeric columns and ensure unique features
    X = data.drop(columns=['Yield_tons_ha', 'Country', 'Unnamed: 0'], errors='ignore')
    X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicate columns
    y = data['Yield_tons_ha']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale and save scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(poly, 'models/poly_features.pkl')
    
    # Save expected columns here (from X, before scaling)
    joblib.dump(X.columns.tolist(), 'models/expected_columns.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"CV R² Scores: {cv_scores.mean():.4f}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/crop_yield_model.pkl')
    
    return model

def predict_yield(input_data, model, scaler, poly):
    """
    Predict crop yield for a single input (dict) or batch (DataFrame).
    Returns predicted yield(s), lower CI, upper CI.
    """
    # Load expected columns from training
    features = joblib.load('models/expected_columns.pkl')
    
    # Convert input to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
        single_input = True
    else:
        df = input_data.copy()
        single_input = False
    
    # Rename input columns to match training
    df = df.rename(columns={
        'Area': 'Country',
        'Item': 'Crop_Type',
        'average_rain_fall_mm_per_year': 'Rainfall_mm',
        'pesticides_tonnes': 'Fertilizer_kg_ha',
        'avg_temp': 'Temperature_C'
    })
    
    # Add Soil_pH if not present
    if 'Soil_pH' not in df.columns:
        df['Soil_pH'] = 6.5
    
    # Validate required columns
    required_cols = ['Crop_Type', 'Year', 'Rainfall_mm', 'Fertilizer_kg_ha', 'Temperature_C', 'Soil_pH']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Preprocess: One-hot encode Crop_Type
    df = pd.get_dummies(df, columns=['Crop_Type'], drop_first=True)
    
    # Align columns with training features
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reindex(columns=features, fill_value=0)
    
    # Validate numeric inputs
    if (df['Temperature_C'] < 0).any() or (df['Temperature_C'] > 50).any():
        raise ValueError("Temperature out of range (0-50°C)")
    if (df['Rainfall_mm'] < 0).any():
        raise ValueError("Rainfall cannot be negative")
    if (df['Fertilizer_kg_ha'] < 0).any():
        raise ValueError("Fertilizer cannot be negative")
    
    # Extract numeric features for polynomial transformation
    numeric_features = ['Rainfall_mm', 'Fertilizer_kg_ha', 'Temperature_C', 'Soil_pH', 'Year']
    X_numeric = df[numeric_features]
    
    # Transform with fitted PolynomialFeatures
    X_poly = poly.transform(X_numeric)
    poly_feature_names = poly.get_feature_names_out(numeric_features)
    df_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)
    
    # Combine original and polynomial features
    df = pd.concat([df.drop(columns=numeric_features), df_poly], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reindex(columns=features, fill_value=0)
    
    # Feature engineering: Interaction term
    if 'Rain_Temp_Interaction' in features:
        df['Rain_Temp_Interaction'] = df['Rainfall_mm'] * df['Temperature_C']
    
    # Scale and predict
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    
    # Compute 95% CI using RandomForest tree predictions
    tree_preds = np.array([tree.predict(scaled) for tree in model.estimators_])
    lower_ci = np.percentile(tree_preds, 2.5, axis=0)
    upper_ci = np.percentile(tree_preds, 97.5, axis=0)
    
    if single_input:
        return preds[0], lower_ci[0], upper_ci[0]
    else:
        return preds, lower_ci, upper_ci