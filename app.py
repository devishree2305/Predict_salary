from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('salary_predictor_with_field.pkl')
encoder = joblib.load('field_label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        years = float(data['years_experience'])
        field = data['field']

        field_encoded = encoder.transform([field])[0]
        input_data = np.array([[years, field_encoded]])

        prediction = model.predict(input_data)
        predicted_salary = max(0, min(2_00_00_000, prediction[0]))
        predicted_yearly = f"₹{int(predicted_salary):,}/year"
        predicted_monthly = f"₹{int(predicted_salary // 12):,}/month"

        return jsonify({
            'prediction': predicted_yearly,
            'monthly': predicted_monthly
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
