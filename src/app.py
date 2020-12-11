#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" demo API
"""
import json
import uvicorn
import joblib
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from typing import Optional, Dict
import numpy as np

from model_architecture import process_tree_for_pred, process_tree
from utils.interpret import interpret_sample_lime
import logging

logger=logging.getLogger()
df = pd.read_csv("/assets/subject/case_study_scoring_raw.csv")
X, y, _ = process_tree(df)
df["fake_opportunity_id"] = df.fake_opportunity_id.astype(str)
df["opportunity_creation_date"] = pd.to_datetime(df.opportunity_creation_date)
df["fake_contact_id"] = df.fake_opportunity_id.astype(str)
df["contact_creation_date"] = pd.to_datetime(df.contact_creation_date)
df["postal_code"] = df.postal_code.astype(str)
df["has_been_recommended"] = df.has_been_recommended.astype(bool)
df = df.drop("opportunity_stage_after_30_days", axis=1)
clf = joblib.load("/assets/models/lgbm.pkl")
le_list = joblib.load("/assets/models/le_list.pkl")



modal_sample =  {'fake_opportunity_id': "id", 
                 'opportunity_creation_date': '2019-10-08 13:52:10',
                 'fake_contact_id': "id",
                 'contact_creation_date': '2015-05-03 01:02:29',
                 'country': "fr",
                 'salesforce_specialty': 'Medecin-generaliste',
                 'contact_status': 'Prospect',
                 'current_opportunity_stage': 'Missed',
                 'opportunity_stage_at_the_time_of_creation': 'Prospection',
                 'opportunity_stage_after_60_days': "60",
                 'opportunity_stage_after_90_days': "90",
                 'previous_max_stage': 'Training',
                 'count_previous_opportunities': 0,
                 'has_mobile_phone': False,
                 'main_competitor': 'FR - PagesJaunes',
                 'has_website': False,
                 'pms': 'LIBERAL',
                 'pms_status': 'no_pms',
                 'count_total_calls': 0,
                 'count_unsuccessful_calls': 0,
                 'count_total_appointments': 0,
                 'count_contacts_converted_last_30d': 3951,
                 'count_contacts_converted_last_30d_per_specialty': 328,
                 'count_contacts_converted_last_30d_per_zipcode': 0,
                 'count_contacts_converted_last_30d_per_specialty_and_zipcode': 0,
                 'gender': 'm',
                 'practitioner_age': 62.0,
                 'years_since_graduation': 35.0,
                 'years_since_last_moving': 2.0,
                 'working_status': 'Temps complet',
                 'days_since_last_inbound_lead_date': 0.0,
                 'days_since_last_congress_lead_date': 0.0,
                 'has_been_recommended': False,
                 'postal_code': "13008.0",
                 'count_clients_with_same_zipcode': 0,
                 'is_city_with_other_clients': True,
                 'count_clients_with_same_zipcode_and_spe': 0,
                 'count_clients_with_same_specialty': 16746,
                 'is_in_dense_area_for_this_cluster': False,
                 'number_of_prospects_in_account': 1,
                 'number_of_clients_in_account': 0}
 
sample_dtypes = df.dtypes


app = FastAPI()
app.mount("/front", StaticFiles(directory="/demo/front"), name="front")




@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Index</title>
        </head>
        <body>
            <h2>Hi User!</h2>
            <h4>no real landing page here please go to either to the "/docs" or "/app" endpoints!</h4>
        </body>
    </html>
    """



@app.get("/app")
async def redirect():
    response = RedirectResponse(url="/front/index.html")
    return response



@app.post("/predict")
async def predict(json_data: Dict=modal_sample):
    """ make a single prediction given jsonlike input data\n
        for a better UX data is initialized as a zero vector that gets updated with posted data
    """
    updated_dict = {**modal_sample, **json_data}
    dfsample = pd.DataFrame([updated_dict], dtype=sample_dtypes.values)
    x, _ = process_tree_for_pred(dfsample, le_list)
    y_pred = clf.predict(x)[0]
    prob = clf.predict_proba(x)[0].tolist()
    return json.dumps({"prediction": int(y_pred),
            "probability": prob,
            "data":updated_dict})



@app.post("/explain", response_class=HTMLResponse)
async def explain(json_data: Dict=modal_sample):
    """ explain a single prediction given jsonlike input data\n
        for a better UX data is initialized as a zero vector that gets updated with posted data
    """
    updated_dict = {**modal_sample, **json_data}
    dfsample = pd.DataFrame([modal_sample], dtype=sample_dtypes.values)
    x, names = process_tree_for_pred(dfsample, le_list)
    return interpret_sample_lime(clf, X, y, x, names)
    
    



@app.post("/batch-predict")
async def batch_predict():
    """ todo compute predictions for a whole batch
    """
    return {"response": "not implemented yet, sry"}




if __name__ == '__main__':
    uvicorn.run(app)