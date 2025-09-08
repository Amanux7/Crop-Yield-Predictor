# Crop Yield Predictor

This project is a crop yield predictor that predicts the yield of a crop based on various factors such as country, year, rainfall, pesticides, and temperature. It provides a user-friendly interface built with Streamlit.

## Features

- **Single Prediction:** Predict crop yield for a single set of inputs.
- **Batch Prediction:** Predict crop yield for a batch of inputs from a CSV file.
- **Dashboard:** Visualize historical data and trends.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

## Project Structure

```
.
├── app.py
├── requirements.txt
├── utils.py
├── data
│   └── yield_df.csv
└── models
    ├── crop_yield_model.pkl
    ├── expected_columns.pkl
    ├── poly_features.pkl
    └── scaler.pkl
```

## Sharing on GitHub and Hugging Face

### GitHub

1.  **Create a new repository on GitHub.**
2.  **Initialize a git repository in your local project directory:**
    ```bash
    git init
    ```
3.  **Add the files to the git repository:**
    ```bash
    git add .
    ```
4.  **Commit the files:**
    ```bash
    git commit -m "Initial commit"
    ```
5.  **Add the GitHub repository as a remote:**
    ```bash
    git remote add origin <repository-url>
    ```
6.  **Push the files to the GitHub repository:**
    ```bash
    git push -u origin master
    ```

### Hugging Face

You can share your Streamlit app on Hugging Face Spaces.

1.  **Create a new Space on Hugging Face.**
2.  **Choose "Streamlit" as the app template.**
3.  **Upload your files (`app.py`, `requirements.txt`, `utils.py`, `data/` and `models/` directories) to the Hugging Face Space.**
4.  **Your app will be deployed automatically.**

You can also use the Hugging Face Hub library to upload your files programmatically.
