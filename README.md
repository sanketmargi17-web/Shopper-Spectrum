# Shopper Spectrum v2.0

## Large Model File

The file `models/similarity.pkl` is not stored in this GitHub repository because it exceeds GitHub's 100 MB file size limit.

When the Product Recommendation module is opened for the first time, the application automatically downloads the model from Google Drive.

### Google Drive Link
https://drive.google.com/file/d/1HXlJxRNt6P7ucwiGZWX-tZx29dVsg7Ss/view?usp=sharing

No manual setup is required.
### Instructions

1. Open the Google Drive folder.
2. Download `similarity.pkl`.
3. Copy the downloaded file into the `models/` folder.
4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run streamlit/app.py
```

After placing the file in the `models` folder, the Product Recommendation module will work correctly.
## Features

- Modern Streamlit dashboard with responsive custom CSS
- KPI cards for revenue, customers, products, and orders
- Interactive Plotly charts for monthly sales, top products, top customers, and country analysis
- Automatic dataset cleaning and creation of `dataset/cleaned_online_retail.csv`
- RFM analysis for customer segmentation
- Customer segmentation using saved `scaler.pkl` and `kmeans.pkl`
- Automatic retraining only when the segmentation artifacts are incompatible
- Product recommendation system using `similarity.pkl`
- Automatic handling for both Pandas DataFrame and NumPy similarity matrices
- Robust error handling for missing files, missing columns, and invalid inputs

## Project Structure

```text
assets/
dataset/
models/
streamlit/
  app.py
  dashboard.py
  segmentation.py
  recommendation.py
  utils.py
styles.css
requirements.txt
README.md
ShopperSpectrum.ipynb
```

## Data Pipeline

1. The app loads `dataset/online_retail.csv`.
2. Invalid rows are removed automatically:
   - missing `CustomerID`
   - cancelled invoices
   - non-positive quantities
   - non-positive unit prices
   - missing or blank descriptions
3. A cleaned dataset is saved to `dataset/cleaned_online_retail.csv`.
4. The cleaned dataset is used for dashboard analytics, RFM segmentation, and product recommendations.

## Models

### Customer Segmentation

- Features: `Recency`, `Frequency`, `Monetary`
- Scaler: `models/scaler.pkl`
- Clustering: `models/kmeans.pkl`
- If the saved artifacts cannot transform or predict against the dataset, the app retrains them automatically and saves the updated versions back into `models/`

### Product Recommendation

- Similarity artifact: `models/similarity.pkl`
- Product list: `models/product_names.pkl`
- The app handles:
  - Pandas DataFrame similarity matrices
  - NumPy array similarity matrices

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run streamlit/app.py
```

## Notes

- Currency values are displayed in Indian Rupees (₹) across the app for presentation consistency.
- The dashboard and segmentation pages respect the sidebar filters.
- The recommendation page always uses the full product catalog so similarity results stay stable.

## Troubleshooting

- If the app reports a missing dataset or model file, confirm that the `dataset/` and `models/` folders are present.
- If you change the raw dataset, rerun the app and the cleaned CSV will be regenerated automatically when needed.
- If a cached model becomes incompatible with the data schema, the app will retrain the segmentation model automatically.
