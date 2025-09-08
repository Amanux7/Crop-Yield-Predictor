import os
from utils import load_and_preprocess_data

# run preprocessing to create and save scaler
X_train_scaled, X_test_scaled, y_train, y_test, cols = load_and_preprocess_data()

import joblib
scaler = joblib.load('models/scaler.pkl')
# import os is intentionally placed before the print as requested
print(scaler)
