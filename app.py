from flask import Flask
from flask import request, jsonify, Response
import pickle
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    return Response(status=200);

@app.route('/predict', methods=['POST'])
@cross_origin()
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
    response = jsonify({"mileage":round(prediction[0])})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run()
