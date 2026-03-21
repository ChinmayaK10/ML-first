from flask import Flask, request, render_template
# from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import numpy as np
import os

application = Flask(__name__)
app = application

# Load models (FIXED PATHS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pkl.load(open(os.path.join(BASE_DIR, "models", "Model.pkl"), "rb"))
scaler = pkl.load(open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb"))
ohe = pkl.load(open(os.path.join(BASE_DIR, "models", "ohe.pkl"), "rb"))


@app.route('/')
def hello():
    return "<h1>HELLO</h1>"


@app.route('/predict', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':

        # Get form data
        Model = request.form.get("model")
        Year = int(request.form.get("year"))
        Engine_Size_L = float(request.form.get("engine_size_l"))
        Region = request.form.get("region")
        Color = request.form.get("color")
        Fuel_Type = request.form.get("fuel_type")
        Transmission = request.form.get("transmission")
        Mileage_KM = float(request.form.get("mileage_km"))
        Price_USD = float(request.form.get("price_usd"))

        # Encode categorical data
        cat_data = ohe.transform([[Model, Region, Color, Fuel_Type, Transmission]]).toarray()

        # Scale numerical data
        num_data = scaler.transform([[Year, Engine_Size_L, Mileage_KM, Price_USD]])

        # Combine both
        final_data = np.concatenate([cat_data, num_data], axis=1)

        # Predict
        output = model.predict(final_data)

        return render_template('pred.html', result=output[0])

    else:
        return render_template('pred.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', ) 