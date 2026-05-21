import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("linear_model.pkl", "rb"))


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# API Route
@app.route("/predict_api", methods=["POST"])
def predict_api():

    data = request.json["data"]

    n_data = pd.DataFrame([data])

    output = model.predict(n_data)

    return jsonify(float(output[0]))


# HTML Form Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    data = [float(x) for x in request.form.values()]

    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX",
        "RM", "AGE", "DIS", "RAD", "TAX",
        "PTRATIO", "B", "LSTAT"
    ]

    n_data = pd.DataFrame([data], columns=columns)

    output = model.predict(n_data)[0]

    return render_template(
        "home.html",
        prediction_text=f"The House price prediction is {output}"
    )


if __name__ == "__main__":
    app.run(debug=True)