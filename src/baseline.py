#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" mock attempt for the baseline described in case study pdf
"""

import click
import joblib
import numpy as np
import pandas as pd

from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from utils.log import logger, logthis
from utils.review import cv_confusion_matrix, plot_learning_curves

extra_args = {"funcname_override": "print"}

np.random.seed(42)


@logthis
def search_baseline(df, shuffle_split_strategy):
    """replicate described baseline with clean nested CV strategy
    being one of stratified shuffle split, repeated  strat kfold, repeated kfold
    """
    X = df.drop(
        ["opportunity_stage_after_30_days", "has_last_inbound_lead"], axis=1
    ).values
    y = df.opportunity_stage_after_30_days.values

    for train_idx, test_idx in shuffle_split_strategy.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        grid = [  # elasticnet
            {
                "C": np.logspace(-2.5, 0.75, 5),
                "l1_ratio": np.linspace(0, 0.5, 5),
                "solver": ["saga"],
                "penalty": ["elasticnet"],
            },
            # ridge
            {"C": np.logspace(-2.5, 0.75, 5)},
        ]
        clf = LogisticRegression(class_weight="balanced")
        lr_grid = GridSearchCV(clf, grid, scoring="roc_auc", cv=5, n_jobs=3)
        lr_grid.fit(X_train, y_train)
        y_pred = lr_grid.best_estimator_.predict(X_test)
        test_score = roc_auc_score(y_test, y_pred)
        logger.info(f"LR parameters {lr_grid.best_params_}", extra=extra_args)
        logger.info(
            f"train auc:  {lr_grid.best_score_:.4f}, test auc: {test_score:.4f}",
            extra=extra_args,
        )


@logthis
def nonnested_search_baseline(df, shuffle_split_strategy):
    """non nested baseline gridsearch"""
    X = df.drop(["opportunity_stage_after_30_days"], axis=1).values
    y = df.opportunity_stage_after_30_days.values
    grid = [  # elasticnet
        {
            "C": np.logspace(-3, 3, 7),
            "l1_ratio": np.linspace(0, 1, 5),
            "solver": ["saga"],
            "penalty": ["elasticnet"],
        },
        # ridge
        {"C": np.logspace(-3, 3, 7)},
    ]
    clf = LogisticRegression()
    lr_grid = GridSearchCV(
        clf,
        grid,
        scoring="roc_auc",
        cv=shuffle_split_strategy,
        return_train_score=True,
        n_jobs=3,
    )
    lr_grid.fit(X, y)
    logger.info(f"LR parameters {lr_grid.best_params_}", extra=extra_args)
    logger.info(f"val auc :  {lr_grid.best_score_:.4f}", extra=extra_args)
    logger.info(f"CV results :  {lr_grid.cv_results_}", extra=extra_args)


@logthis
def train_and_serialize(clf, X, y):
    """train and serialize simple estimator"""
    clf.fit(X, y)
    joblib.dump(clf, "assets/models/baseline.pkl")


@logthis
def most_incorrect(clf, X, y, shuffle_split_strategy, keep_top=50):
    """compute most incorrect predictions given clf and CV strategy
    returns list of incorrect predictions, sorted by decreasing model confidence
    NB: this should ideally be done w/ trained estimator in a formal test setting
    """
    incorrect = []
    for train_idx, test_idx in shuffle_split_strategy.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]
        # aggregate wrong predictions
        wrong = [
            (x, yp, y, p)
            for x, yp, y, p in list(zip(X_test, preds, y_test, probas))
            if yp != y
        ]
        # filter to get wrong predictions sorted by
        wrong.sort(key=lambda tup: abs(tup[-1] - 0.5), reverse=True)
        wrong = wrong[:keep_top]
        incorrect.append(wrong)
    return incorrect


@logthis
def generate_incorrect_report(incorrect_list, columns):
    """given list of incorrect predictions generate profile report"""
    # make incorrect df to find patterns
    data = np.vstack(
        [
            np.hstack([tup[2], tup[0], tup[-1]])
            for list_ in incorrect_list
            for tup in list_
        ]
    )
    incorrect_df = pd.DataFrame(data, columns=columns.tolist() + ["p"])
    profile = ProfileReport(incorrect_df, explorative=True)
    profile.to_file("assets/wip/report_incorrect.html")


@click.command()
@click.option("--search", is_flag=True)
@click.option("--diagnose", is_flag=True)
@click.option("--train", is_flag=True)
def main(**kwargs):
    df = pd.read_csv("assets/subject/case_study_scoring_clean.csv", sep=";")
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    X = df.drop(["opportunity_stage_after_30_days"], axis=1).values
    y = df.opportunity_stage_after_30_days.values
    clf = LogisticRegression(
        C=0.1,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.25,
        class_weight="balanced",
    )
    if kwargs["search"]:
        search_baseline(df, sss)
    if kwargs["diagnose"]:
        plot_learning_curves(
            clf,
            make_scorer(roc_auc_score),
            X,
            y,
            filename="assets/ouput/LC_baseline.png",
            cv=sss,
            n_jobs=3,
        )
        cv_confusion_matrix(
            clf, X, y, sss, filename="assets/output/CM_baseline.png", normalize=False
        )
        cv_confusion_matrix(
            clf, X, y, sss, filename="assets/output/CMprec_baseline.png", normalize=True
        )
    if kwargs["train"]:
        train_and_serialize(clf, X, y)
        print("train")


if __name__ == "__main__":
    main()
