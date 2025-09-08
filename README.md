# Crop Yield Prediction System

A machine learning-based web application for forecasting crop yields (wheat, rice, maize) to support food security, resource optimization for farmers, NGOs, and policymakers. Built with Python, scikit-learn, and Streamlit.

## Purpose
This ML app uses a RandomForestRegressor (optimized from Linear Regression) to predict crop yields based on features like rainfall, temperature, pesticides, year, country, and crop type. It helps users like farmers (e.g., Raj), NGOs (e.g., Maria), and researchers (e.g., Dr. Lee) make data-driven decisions for sustainable agriculture.

## Key Features
- **Data Preprocessing**: Handles Kaggle dataset with normalization (e.g., pesticides to kg/ha), one-hot encoding for countries/crops, polynomial features (degree=2), and scaling.
- **Model Training/Evaluation**: RandomForest with cross-validation; targets R² >0.8, MSE <10% of average yield (~0.35 tons/ha).
- **Web UI**: Streamlit dashboard for single/batch predictions with 95% confidence intervals (CI), visualizations (trends, scatters, bars), alerts (low-yield warnings), and report downloads.
- **Performance**: Predictions <2s; scalable to 1,000 users/day.
- **Security**: HTTPS-ready for deployment.

### In-Scope
- Data prep, model training/evaluation, web UI, visualizations, deployment.

### Out-of-Scope (Future Phases)
- Real-time satellite integration, mobile app, advanced maps (e.g., folium).

## Assumptions
- Focus on wheat, rice, maize; uses open Kaggle dataset (~21,268 rows).
- Basic internet access; no real-time data.

## Goals Achieved
- R² ~0.97 (exceeds >0.8).
- MSE ~0.85 (under 10% of average yield).
- Predictions with CI; <2s latency.

## Project Structure
Crop Yield Prediction System/
├── crop_yield_env/          # Virtual environment
├── data/
│   └── yield_df.csv         # Kaggle dataset (~21k rows)
├── models/
│   ├── crop_yield_model.pkl # Trained RandomForest
│   ├── scaler.pkl           # StandardScaler
│   ├── poly_features.pkl    # PolynomialFeatures (degree=2)
│   └── expected_columns.pkl # Feature names post-preprocessing
├── src/                     # Optional: Modular code (expandable)
├── app.py                   # Streamlit UI for predictions/dashboard
├── utils.py                 # Core logic: preprocess, train, predict
├── test.py                  # Testing script (train + sample predict)
├── requirements.txt         # Dependencies (scikit-learn, streamlit, etc.)
└── README.md                # This file
text## Installation
1. **Clone/Setup**:
   - GitHub: `git clone https://github.com/yourusername/crop-yield-prediction.git`
   - Create/activate virtual env: `python -m venv crop_yield_env` then `source crop_yield_env/bin/activate` (Linux/macOS) or `crop_yield_env\Scripts\activate` (Windows).

2. **Install Dependencies**:
pip install -r requirements.txt
textContents of `requirements.txt`:
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
streamlit==1.25.0
joblib==1.3.1
matplotlib==3.7.2
text3. **Prepare Data/Models**:
- Place `yield_df.csv` in `data/` (download from Kaggle: search "crop yield prediction dataset").
- Run training: `python test.py` (creates models/ artifacts; prints R²/MSE).

## Usage
1. **Run the App**:
streamlit run app.py
text- Opens at http://localhost:8501.

2. **Single Prediction** (for farmers like Raj):
- Select country/crop, enter year/rainfall/pesticides/temp.
- Get yield (tons/ha) with 95% CI and alerts (e.g., "Below average—optimize resources").

3. **Batch Prediction** (for NGOs like Maria):
- Upload CSV with columns: Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.
- Download results with predictions/CI.

4. **Dashboard** (for researchers like Dr. Lee):
- View trends (yield over years), scatters (yield vs. temp), top countries bar charts, and tables.

Example Input (India Wheat 2023):
- Country: India
- Crop: Wheat
- Year: 2023
- Rainfall: 1102.0 mm/year
- Pesticides: 61702.0 tonnes
- Temp: 25.65°C
- Expected Output: ~3.5 tons/ha (95% CI: 2.8–4.2).

## Technical Specs
- **Language**: Python 3.x
- **ML**: scikit-learn (RandomForestRegressor, n_estimators=200)
- **UI**: Streamlit
- **Data**: Pandas, NumPy
- **Deployment**: Ready for Heroku/AWS (add Procfile: `web: streamlit run app.py --server.port=$PORT`).
- **Risks Mitigated**: Data bias (diverse countries/crops, normalization); overfitting (CV, poly features).

## Current Status
- **Phase 1 Complete**: Data prep, model training (R² ~0.97, MSE ~0.85).
- **Phase 2 Complete**: Streamlit UI with predictions/dashboard.
- **Phase 3 Pending**: Deployment, maps (folium).
- **Phase 4 Pending**: Mobile app, real-time data.

## Contributing
1. Fork the repo.
2. Create branch: `git checkout -b feature-branch`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push: `git push origin feature-branch`.
5. Open PR.

## License
MIT License (or your choice—add LICENSE file).

## Acknowledgments
- Dataset: Kaggle Crop Yield Prediction.
- Inspired by food security initiatives (FAO, World Bank).

For issues/bugs: Open a GitHub issue. Questions? Contact amanux66@gmail.com
.
