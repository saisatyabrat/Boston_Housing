import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("linear_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():

    data = request.json["data"]

    # Convert into dataframe
    n_data = pd.DataFrame([data])

    output = model.predict(n_data)

    return jsonify(float(output[0]))

if __name__ == "__main__":
    app.run(debug=True)