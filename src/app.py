#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" demo API
"""
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from typing import Optional, Dict

from utils.interpret import interpret_sample_lime

df = pd.read_csv("/assets/subject/case_study_scoring_clean.csv", sep=";")
y = df["opportunity_stage_after_30_days"].values
df = df.drop(["opportunity_stage_after_30_days"],axis=1)
colnames = df.columns.values
basedict = dict(zip(colnames, [0]*len(colnames)))
clf = joblib.load("/assets/models/baseline.pkl")
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Index</title>
        </head>
        <body>
            <h2>Hi User!</h2>
            <h4>no real landing page here please go to the "/docs" endpoint!</h4>
        </body>
    </html>
    """


@app.post("/predict")
async def predict(json_data: Dict={"contact_status_churner": 0, "contact_status_prospect": 1}):
    """ make a single prediction given jsonlike input data\n
        for a better UX data is initialized as a zero vector that gets updated with posted data
    """
    updated_dict = {**basedict, **json_data}
    x = pd.Series(updated_dict).values.reshape(1,-1)
    y_pred = clf.predict(x)[0]
    prob = clf.predict_proba(x)[0].tolist()
    return {"prediction": int(y_pred),
            "probability": prob,
            "data":updated_dict}


 


@app.post("/explain", response_class=HTMLResponse)
async def explain(json_data: Dict={"contact_status_churner": 0, "contact_status_prospect": 1}):
    """ explain a single prediction given jsonlike input data\n
        for a better UX data is initialized as a zero vector that gets updated with posted data
    """
    updated_dict = {**basedict, **json_data}
    x = pd.Series(updated_dict)#.values.reshape(1,-1)
    interpret_sample_lime(clf, df.values, y, x)
    
    





@app.post("/batch-predict")
async def batch_predict():
    """ todo compute predictions for a whole batch
    """
    return {"response": "not implemented yet, sry"}




if __name__ == '__main__':
    uvicorn.run(app)