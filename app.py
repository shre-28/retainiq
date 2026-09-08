from flask import Flask, render_template, request
import joblib
import pandas as pd
import os


# --------------------------------------------------
# Create Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Find the trained model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "xgboost_churn_model.pkl"),
    os.path.join(BASE_DIR, "notebooks", "xgboost_churn_model.pkl")
]


def load_model():
    """
    Load the trained XGBoost pipeline.
    Checks both root folder and notebooks folder.
    """

    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path)

    raise FileNotFoundError(
        "xgboost_churn_model.pkl was not found. "
        "Please upload the model file to the GitHub repository."
    )


# Model will be loaded only when prediction is requested
model = None


def get_model():
    global model

    if model is None:
        model = load_model()

    return model


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ------------------------------------------
        # Get customer information from form
        # ------------------------------------------

        credit_score = float(request.form["CreditScore"])
        geography = request.form["Geography"]
        gender = request.form["Gender"]
        age = float(request.form["Age"])
        tenure = float(request.form["Tenure"])
        balance = float(request.form["Balance"])
        num_products = float(request.form["NumOfProducts"])
        has_credit_card = float(request.form["HasCrCard"])
        is_active_member = float(request.form["IsActiveMember"])
        estimated_salary = float(request.form["EstimatedSalary"])


        # ------------------------------------------
        # Create DataFrame
        # ------------------------------------------

        customer_data = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_credit_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary
        }])


        # ------------------------------------------
        # Load trained model
        # ------------------------------------------

        trained_model = get_model()


        # ------------------------------------------
        # Make prediction
        # ------------------------------------------

        prediction = trained_model.predict(customer_data)[0]


        # ------------------------------------------
        # Get churn probability
        # ------------------------------------------

        probability = trained_model.predict_proba(customer_data)[0][1]

        churn_probability = round(probability * 100, 2)


        # ------------------------------------------
        # Prediction result
        # ------------------------------------------

        if prediction == 1:
            result = "Customer is likely to churn"
        else:
            result = "Customer is likely to stay"


        # ------------------------------------------
        # Send result to frontend
        # ------------------------------------------

        return render_template(
            "index.html",
            prediction=result,
            probability=churn_probability
        )


    except Exception as e:

        # Show error on webpage instead of crashing silently
        return render_template(
            "index.html",
            prediction="Prediction Error",
            probability=None,
            error=str(e)
        )


# --------------------------------------------------
# Run application locally
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )