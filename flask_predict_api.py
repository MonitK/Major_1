import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

#reading the binaryversion of python script
with open('E:/ATISHAY/C/major 1/random forest basic app/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

#defining a method that can be used to accept input and used that as data set on trained model
#using swagger to provide a predefined UI
@app.route('/predict')
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    
    
    
    