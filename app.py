# app.py
from flask import Flask, request, jsonify
from model import ModelLoader
from preprocessing import preprocess_data

app = Flask(__name__)

# Load model once when Flask app starts
model_loader = ModelLoader(model_path='path_to_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Make prediction
        prediction = model_loader.predict(processed_data)
        
        # Return prediction in a JSON response
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
