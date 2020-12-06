#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" minimal testing suite for ml models
"""
#todo handle python path
import joblib
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def load_test_tuple():
    """ load model and example
    """
    model = joblib.load("assets/models/simple.pkl")
    df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")\
    .drop(["opportunity_stage_after_30_days"],axis=1)
    example = df.sample(1).values
    return model, example




def test_output_shape(load_test_tuple):
    """ test if model outputs binary classification probem
    """
    model, example = load_test_tuple
    print(model.predict(example))
    print(model.predict_proba(example))
    assert model.predict(example).shape == (1,)
    assert model.predict_proba(example).shape ==(1, 2)


def test_output_range(load_test_tuple):
    """ test if model outputs correctly a probability
    """
    model, example = load_test_tuple
    print(model.predict(example))
    print(model.predict_proba(example))
    assert np.sum(model.predict_proba(example)) == 1


