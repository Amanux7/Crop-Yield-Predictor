# Updated test.py: Remove features from load_and_preprocess_data return, since it's no longer returned (saved inside function).

from utils import load_and_preprocess_data, train_model, predict_yield
import joblib

# Load and preprocess (this now saves expected_columns.pkl)
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train and save model
model = train_model(X_train, y_train, X_test, y_test)

# Load for testing (optional)
scaler = joblib.load('models/scaler.pkl')
poly = joblib.load('models/poly_features.pkl')

# Test prediction (example)
sample_input = {
    'Area': 'India',
    'Item': 'Wheat',
    'Year': 2023,
    'average_rain_fall_mm_per_year': 650.0,
    'pesticides_tonnes': 2.0,
    'avg_temp': 20.0
}
yield_pred, lower_ci, upper_ci = predict_yield(sample_input, model, scaler, poly)
print(f"Predicted Yield: {yield_pred:.2f} tons/ha (CI: {lower_ci:.2f} - {upper_ci:.2f})")