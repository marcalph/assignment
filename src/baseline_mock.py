#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" mock attempt for the baseline described in case study pdf
"""
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, train_test_split, StratifiedShuffleSplit, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import joblib

from utils.log import logger, logthis
extra_args = { "funcname_override" : "print"}

np.random.seed(42)

@logthis
def search_baseline(df, shuffle_split_strategy):
    """ replicate described baseline with clean nested CV strategy
        being one of stratified shuffle split, repeated  strat kfold, repeated kfold 
    """
    X = df.drop(["opportunity_stage_after_30_days", "has_last_inbound_lead"],axis=1).values
    y = df.opportunity_stage_after_30_days.values

    for train_idx, test_idx in shuffle_split_strategy.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        grid=[# elasticnet
              {"C":np.logspace(-3,3,7), "l1_ratio":np.linspace(0,1,5), "solver":["saga"] , "penalty":["elasticnet"]},
              # ridge
              {"C":np.logspace(-3,3,7)}
        ]
        clf = LogisticRegression(class_weight="balanced")
        lr_grid = GridSearchCV(clf, grid, scoring="roc_auc", cv=5, n_jobs=3)
        lr_grid.fit(X_train, y_train)
        logger.info(f"LR parameters {lr_grid.best_params_}", extra=extra_args)
        logger.info(f"train auc :  {lr_grid.best_score_:.4f}, test auc : {roc_auc_score(y_test, lr_grid.best_estimator_.predict(X_test)):.4f}", extra=extra_args)



@logthis
def nonnested_search_baseline(df, shuffle_split_strategy):
    """ non nested baseline gridsearch
    """
    X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
    y = df.opportunity_stage_after_30_days.values
    grid=[# elasticnet
              {"C":np.logspace(-3,3,7), "l1_ratio":np.linspace(0,1,5), "solver":["saga"] , "penalty":["elasticnet"]},
              # ridge
              {"C":np.logspace(-3,3,7)}
        ]
    clf = LogisticRegression()
    lr_grid = GridSearchCV(clf, grid, 
                           scoring="roc_auc",
                           cv=shuffle_split_strategy,
                           return_train_score=True,
                           n_jobs=3)
    lr_grid.fit(X, y)
    logger.info(f"LR parameters {lr_grid.best_params_}", extra=extra_args)
    logger.info(f"val auc :  {lr_grid.best_score_:.4f}", extra=extra_args)
    logger.info(f"CV results :  {lr_grid.cv_results_}", extra=extra_args)




@logthis
def train_simple(df):
    """ train and serialize simple estimator
    """
    X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
    y = df.opportunity_stage_after_30_days.values
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X, y)
    joblib.dump(clf, 'assets/models/simple.pkl')



if __name__ == "__main__":
    df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
    sss = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    search_baseline(df, sss)
