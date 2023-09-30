from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
data = pd.read_csv("Cleaned_data.csv")

@app.route('/')
def index():
    NdsCH = sorted(data['NdsCH'].unique())
    NdssC = sorted(data['NdssC'].unique())
    return render_template('index.html',NdsCH=NdsCH,NdssC=NdssC)

@app.route('/predict',methods=['POST'])
def predict():
    CIC0 = float(request.form.get('CIC0'))
    SM1_Dz = float(request.form.get('SM1_Dz(Z)'))
    GATS1i = float(request.form.get('GATS1i'))
    NdsCH = float(request.form.get('NdsCH'))
    NdssC = int(request.form.get('NdssC'))
    MLOGP = float(request.form.get('MLOGP'))
    prediction = model.predict(pd.DataFrame([[CIC0,SM1_Dz,GATS1i,NdsCH,NdssC,MLOGP]],columns=['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP']))
    return str(prediction)

if __name__=="__main__":
    app.run(debug=True)