from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'🏡 Predicted House Price: ${output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
