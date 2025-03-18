from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON data from frontend
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Convert list to NumPy array

        # Ensure the correct number of features
        if features.shape[1] != 13:
            return jsonify({"error": f"Expected 13 features, but got {features.shape[1]}"})

        # Scale the input data
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Return the prediction as JSON
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
