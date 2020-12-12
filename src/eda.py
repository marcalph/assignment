#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" simple exploratory analysis
"""
import os

import pandas as pd
from pandas_profiling import ProfileReport

from utils.log import logthis

extra_args = {
    "funcname_override": "print",
    "filename_override": os.path.basename(__file__),
}


df = pd.read_csv("assets/subject/case_study_scoring_raw.csv")
df["fake_opportunity_id"] = df.fake_opportunity_id.astype(str)
df["opportunity_creation_date"] = pd.to_datetime(df.opportunity_creation_date)
df["fake_contact_id"] = df.fake_opportunity_id.astype(str)
df["contact_creation_date"] = pd.to_datetime(df.contact_creation_date)
df["postal_code"] = df.postal_code.astype(str)


cleandf = pd.read_csv("assets/subject/case_study_scoring_clean.csv", sep=";")


@logthis
def generate_report(df, output_filename):
    """generate simple EDA report given pandas df"""
    profile = ProfileReport(df, explorative=True)
    profile.to_file(output_filename)


generate_report(df, "assets/output/reports/report_raw.html")
generate_report(cleandf, "assets/output/reports/report_clean.html")
