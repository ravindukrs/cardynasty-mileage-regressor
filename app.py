from flask import Flask
from flask import request, jsonify, Response
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    return Response(status=200);

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    body = data.get("body")
    year = data.get("year")
    loaded_model = pickle.load(open('svrmodel.sav', 'rb'))
    le_X_0 = pickle.load(open('label_encoder.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))

    mydata = np.array([[body, year]])
    mydata[:, 0] = le_X_0.transform(mydata[:, 0])
    mydata = scaler.transform(mydata)
    prediction = loaded_model.predict(mydata)
    response = jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return jsonify({"mileage":round(prediction[0])})


if __name__ == '__main__':
    app.run()
