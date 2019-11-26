#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
from flask import Flask, request, jsonify
from flasgger import Swagger
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
#from flask_json import FlaskJSON, JsonError, json_response, as_json
app = Flask(__name__)
swagger = Swagger(app)
#FlaskJSON(app)

# loading the dataset
iris = load_iris()
X = iris.data
y = iris.target
    
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)
    

@app.route('/input',methods=["POST"])
def input_data():
    """Input for dataset and model
    ---
    tags:
      - "Model and Dataset(Dataset is chosen IRIS by default for reference)"
    parameters:
      - name: "Model"
        in: "query"
        description: "Chosse the model on which you want to execute"
        required: true
        type: "array"
        items:
          type: "string"
          enum:
          - "Support Vector Machines"
          - "Logistic Regression"
          - "Naive Bayes Classifier"
          - "Nearest Neighbor"
          - "Decision Trees"
          - "Random Forest"
          default: "Random Forest"
        collectionFormat: "multi"
    responses:
      500:
        description: Error The input is wrong!
      200:
        description: An api where ML is easy!
    """
    #input_model = pd.read_csv(request.files.get("Dataset"), header=None)
    model = request.args.get("Model")
    #print(str(input_model))
    #print()
    #prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    #print(prediction)
    
    if model=="Logistic Regression":
        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        predicted = classifier.predict(X_test)
        # Check accuracy
        print(accuracy_score(predicted, y_test))
    elif model=="Random Forest":    
        # Random Forest Classifier
        from sklearn.ensemble import RandomForestClassifier
        # Build the model
        classifier = RandomForestClassifier(n_estimators=10)
        # Train the classifier
        classifier.fit(X_train, y_train)
        # Predictions
        predicted = classifier.predict(X_test)
        # Check accuracy
        print(accuracy_score(predicted, y_test))
    elif model=="Support Vector Machines":    
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        predicted = classifier.predict(X_test)
        # Check accuracy
        print(accuracy_score(predicted, y_test))
    elif model=="Naive Bayes Classifier":    
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        predicted = classifier.predict(X_test)
        # Check accuracy
        print(accuracy_score(predicted, y_test))
    elif model=="Nearest Neighbor":    
        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        predicted = classifier.predict(X_test)
        # Check accuracy
        print(accuracy_score(predicted, y_test)) 
    else:   
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        predicted = classifier.predict(X_test)
        # Check accuracy
        print(accuracy_score(predicted, y_test)) 
    import pickle
    with open('E:/ATISHAY/C/major 1/random forest basic app/rf.pkl', 'wb') as model_pkl:
        pickle.dump(classifier, model_pkl, protocol=2)
    return str(model+" Succesfully built")

@app.route('/predict')
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---  
    tags:
      - "Input Values for prediction"
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
    responses:
      500:
        description: Error The input is wrong!
      200:
        description: An api where ML is easy!
    """
    with open('E:/ATISHAY/C/major 1/random forest basic app/rf.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    s_length = float(request.args.get("s_length"))
    s_width = float(request.args.get("s_width"))
    p_length = float(request.args.get("p_length"))
    p_width = float(request.args.get("p_width"))
    
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    #print(prediction)
    return str(prediction)
    
@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses:
      500:
        description: Error The input is wrong!
      200:
        description: An api where ML is easy!
    """
    with open('E:/ATISHAY/C/major 1/random forest basic app/rf.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)    