# Flask Libraries
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create Flask app

app = Flask(__name__)

# Load the pickle model
import pickle

model = pickle.load(open("model.pkl", "rb"))

# Home page for app


@app.route("/")
def home():
    return render_template("mlmodel.html")


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Assuming you have a form with input fields in your HTML
        float_features = [float(x) for x in request.form.values()]

        # Check if the number of features matches the model's expectations
        if len(float_features) != 7:
            return render_template("mlmodel.html", prediction_text="Invalid number of features. Please provide all required features.")

        features = [np.array(float_features)]

        # Now 'float_features' is a list of float values obtained from the form data
        # You can use this list for making predictions with your machine learning model

        # Example: Assuming 'model' is your trained model
        predictions = model.predict(features)

        # Assuming your model returns a single prediction
        prediction = predictions[0]

        # Create a text based on the prediction
        prediction_text = "Person is Alive" if predictions[0] == 0 else "Person is Not Alive"

        return render_template("mlmodel.html", prediction_text=prediction_text)

    # If the request method is GET, render the form page
    return render_template("mlmodel.html")

if __name__ == "__main__":
    app.run(debug=True)