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
from typing import Optional, Dict


df = pd.read_csv("/assets/assignment/case_study_scoring_clean.csv", sep=";")\
    .drop(["opportunity_stage_after_30_days"],axis=1)
colnames = df.columns.values
basedict = dict(zip(colnames, [0]*len(colnames)))
clf = joblib.load("/assets/models/simple.pkl")
app = FastAPI()


@app.get("/")
def read_root():
    return {"README": "Hello <user>, no UI is available at the moment please go to the default '/docs' route"}


@app.post("/predict")
def predict(json_data: Dict={"contact_status_churner": 0, "contact_status_prospect": 1, "text2":0}):
    """ make a single prediction given jsonlike input data\n
        for a better UX data is initialized as a zero vector that gets updated with posted data
    """
    updated_dict = {**basedict, **json_data}
    x = pd.Series(updated_dict).values.reshape(1,-1)
    y_pred = clf.predict(x)[0]
    prob = clf.predict_proba(x)[0].tolist()
    text = json_data.get('text')
    return {"prediction": int(y_pred),
            "probability": prob,
            "data":updated_dict}


@app.post("/batch-predict")
def batch_predict():
    """ todo compute predictions for a whole batch
    """
    return {"response": "not implemented yet, sry"}




if __name__ == '__main__':
    uvicorn.run(app)