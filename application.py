import pickle
from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge and regressor and standardscaler pickle

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))



@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        results_rounded = round(result[0],1)

        if result[0] >= 80:
            prediction = f"Chance of Fire, with prediction of: {results_rounded}%"
        else:
            prediction = f"No chance of Fire, with the prediction of: {results_rounded}%"

        return render_template('home.html',results=prediction)

        
    else:
        return render_template('home.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)