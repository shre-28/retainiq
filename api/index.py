import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Flask App Configuration
app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_ROOT, 'templates'), 
            static_folder=os.path.join(PROJECT_ROOT, 'static'))

# Load XGBoost Model
MODEL_PATH = os.path.join(PROJECT_ROOT, 'notebooks - Copy', 'xgboost_churn_model.pkl')

model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"App running, template error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        input_df = pd.DataFrame([data])
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        return jsonify({
            'status': 'success',
            'churn_prediction': int(prediction),
            'churn_probability': float(prob) if prob is not None else None
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
