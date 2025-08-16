from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__, static_folder="static")

# Load dataset
url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
df = pd.read_csv(url)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Route for home page
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([[data['N'], data['P'], data['K'], data['temperature'],
                            data['humidity'], data['ph'], data['rainfall']]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return jsonify({'crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
