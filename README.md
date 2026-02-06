# California Housing Price Prediction

A regression project that predicts California district median house values from socioeconomic and geographic features. This repo includes a modeling notebook and a small [web demo](https://californiahousepricing-c6u8.onrender.com/) for interactive predictions.

## What this project does

The goal is to predict `median_house_value` using features like median income, housing age, population, and latitude and longitude. One important detail is that the target is capped at `5.0` in this dataset, so evaluation can look artificially better if you do not account for that. In the notebook, I compare results on the full dataset and on an uncapped subset for a more realistic view of continuous prediction performance.

## Dataset

This project uses the California Housing dataset that is commonly accessed via `sklearn.datasets.fetch_california_housing`.

## Approach

### 1. Baselines and diagnostics
* Start with a simple baseline to set expectations.
* Run multicollinearity diagnostics using VIF to understand feature relationships before trusting coefficients.

### 2. Modeling
I compare a linear baseline with a stronger nonlinear model.

* OLS regression with feature standardization as an interpretable baseline
* XGBoost regressor for nonlinear interactions and improved accuracy
* Early stopping and cross validation to manage overfitting risk

### 3. Evaluation
Metrics:
* R squared
* RMSE

I report results on the uncapped subset to better reflect real continuous prediction quality.

## Results

On the uncapped subset
* OLS with standardization achieved R squared about 0.554 and RMSE about 0.654
* XGBoost achieved R squared about 0.818 and RMSE about 0.418 on the held out test set
* 5 fold cross validation for XGBoost reached R squared about 0.832 with CV RMSE about 0.398

These numbers may vary slightly based on random seed, train test split, and hyperparameters.

## Repository structure

- `docs/` static site assets
  - `index.html` notebook view
- `templates/` HTML templates for the demo
  - `home.html`
- `Linear_Regression_ML.ipynb` notebook with analysis and training
- `app.py` web app for running predictions
- `xgb_booster.pkl` trained model artifact
- `requirements.txt` Python dependencies
- `Procfile` deployment entrypoint
- `LICENSE` project license

## How to run locally

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
