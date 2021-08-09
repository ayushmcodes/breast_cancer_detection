from operator import mod
from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('breast_cancer_detection.pkl','rb'))


@app.route('/')
def home():    
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        feat1=float(request.form['a'])
        feat2=float(request.form['b'])
        feat3=float(request.form['c'])
        feat4=float(request.form['d'])
        feat5=float(request.form['e'])
        input_features=[]
        input_features.append(feat1)
        input_features.append(feat2)
        input_features.append(feat3)
        input_features.append(feat4)
        input_features.append(feat5)
        features_values=[np.array(input_features)]
        features_name=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
        df=pd.DataFrame(features_values,columns=features_name)
        print(df)
        output=model.predict(df)
        print(output)
        if output==1:
            cancer_type='Melanin'
        else:
            cancer_type='Beganin'
        print(cancer_type)
        return render_template('index.html',output_text='Patient has {}'.format(cancer_type))

if __name__=='__main__':
    app.run(debug=True)#debug=True means run in debug mode