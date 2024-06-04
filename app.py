import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

app=Flask(__name__)
Classification_model=pickle.load(open('LogisticRegression.pkl','rb'))
Regression_model=pickle.load(open('AdaBoostRegressor.pkl','rb'))


@app.route('/predict_api/Classification',methods=['POST'])
def predict_Classification():
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    print(new_data)
    output=(Classification_model.predict(new_data))[0]
    if output == 1:
        return {"Prediction_Result":"Fire"}
    else:
        return{"Prediction_Result":"Not Fire"}


@app.route('/predict_api/Regression',methods=['POST'])
def predict_Regression():
    data=request.json['data']
    new_data=[list(data.values())]
    output=(Regression_model.predict(new_data))[0]
    return{"Prediction_Result":output}



if __name__=="__main__":
    app.run(debug=True)