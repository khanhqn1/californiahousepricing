import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_ORDER = ["MedInc", "HouseAge", "AveRooms", "Population", "AveOccup", "Latitude", "Longitude"]

app = Flask(__name__)
# Load the pre-trained XGBoost model
xgbmodel = pickle.load(open('xgb_booster.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    x = np.array([data[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    print(x)
    dmat = xgb.DMatrix(x, feature_names=FEATURE_ORDER)
    pred = xgbmodel.predict(dmat)
    print(pred[0])
    return jsonify(float(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)