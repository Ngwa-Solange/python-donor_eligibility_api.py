from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("donor_eligibility_model.pkl")

# Label encoders (match training)
gender_map = {'F': 0, 'M': 1}
blood_map = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}

app = Flask(__name__)

@app.route('/')
def home():
    return "Donor Eligibility Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and encode inputs
    try:
        age = float(data['donor_age'])
        hemoglobin = float(data['hemoglobin_g_dl'])
        volume = float(data['collection_volume_ml'])
        gender = gender_map[data['donor_gender'].upper()]
        blood = blood_map[data['blood_type'].upper()]
    except Exception as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    # Prepare input and predict
    input_features = np.array([[age, hemoglobin, volume, gender, blood]])
    prediction = model.predict(input_features)[0]

    return jsonify({'eligible': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)