
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import flask
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    
    Term = flask.request.form['Term']
    NoEmp = flask.request.form['NoEmp']
    NewExist = flask.request.form['NewExist']
    CreateJob = flask.request.form['CreateJob']
    RetainedJob = flask.request.form['RetainedJob']
    FranchiseCode = flask.request.form['FranchiseCode']
    UrbanRural = flask.request.form['UrbanRural']
    LowDoc = flask.request.form['LowDoc']
    ChgOffPrinGr = flask.request.form['ChgOffPrinGr']
    SBA_Appv = flask.request.form['SBA_Appv']
    DaysforDibursement = flask.request.form['DaysforDibursement']
 
    input_variables = pd.DataFrame([[Term,NoEmp,NewExist,CreateJob,RetainedJob,FranchiseCode,UrbanRural,LowDoc,ChgOffPrinGr,SBA_Appv,DaysforDibursement]],columns=['Term','NoEmp','NewExist','CreateJob','RetainedJob', 'FranchiseCode','UrbanRural','LowDoc','ChgOffPrinGr','SBA_Appv','DaysforDibursement'],dtype=float)
    prediction = model.predict(input_variables)[0]
    return render_template('index.html',prediction_text="Defaulter" if prediction==0 else "Not Defaulter")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)