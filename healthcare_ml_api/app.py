from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([
            data['age'],
            data['gender'],
            data['blood_pressure'],
            data['glucose_level'],
            data['bmi'],
            data['heart_rate'],
            data['previous_admissions'],
            data['chronic_disease'],
            data['smoking_status']
        ]).reshape(1, -1)

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        if prediction == 1:
            risk = "High Risk of Readmission"
        else:
            risk = "Low Risk of Readmission"

        response = {
            "success": True,
            "prediction": int(prediction),
            "risk_level": risk,
            "probability": round(float(probability), 2)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
