from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import pandas as pd
import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library

from keras.models import load_model
from keras.utils import np_utils


app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

@app.route("/")
def home():
    html = "<h3>Sklearn Prediction Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
    """Performs an sklearn prediction
        
        input looks like:
        {"SL":0.07471338, "SW":0.09794497, "PL":0.02951407, "PW": 0.01150299}
        
        result looks like:
        { "prediction": [ <val> ] }
        
        """
    
    # Logging the input payload
    data = request.json
    LOG.info("JSON payload:" + str(data) + '\n')


    X_test = [[data['SL'], data['SW'], data['PL'], data['PW']]]
    prediction=model.predict(X_test)
    predict_label=np.argmax(prediction,axis=1)
    d = {}
    d['prediction'] = str(predict_label[0])
    # TO DO:  Log the output prediction value
    LOG.info("Predicted label:" + str(predict_label[0]) + '\n')
    #return jsonify({'prediction': prediction})
    return str(d)

if __name__ == "__main__":
    model = load_model('model.h5')
    app.run(host='0.0.0.0', port=80, debug=True) # specify port=80
