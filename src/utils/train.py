#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" training utils
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, make_scorer

from log import logger, logthis
extra_args = { "funcname_override" : "print"}


@logthis
def plot_learning_curve(estimator, scoring, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """ Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       scoring=scoring,
                       return_times=True)

    # learning curves
    sns.set_theme()
    sizes = np.tile(train_sizes,10)
    sizes.sort()
    df = pd.concat([pd.DataFrame({"Sample size":sizes, "Score":train_scores.reshape(-1), "Set":"train"}),
                    pd.DataFrame({"Sample size":sizes, "Score":test_scores.reshape(-1), "Set":"CV"})])
    sns.lineplot(data=df, 
                 x="Sample size",
                 y="Score",
                 hue="Set", 
                 ci=95, 
                 marker="o",
                 ax=axes[0]).set_title("LC")

    # n_samples vs fit_times
    sns.lineplot(data=pd.DataFrame({"Sample size":sizes, "Fitting time":fit_times.reshape(-1)}),
                 x="Sample size", 
                 y="Fitting time", 
                 ci=95, 
                 marker="o",
                 ax=axes[1]).set_title("Scalability")



@logthis
def cv_classification_report():
    """ generate a cross validated classification report
    """    
    pass



@logthis
def cv_confusion_matrix():
    """ compute a cross validated confusion matrix
    """
    pass



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = LogisticRegression(C=100, class_weight="balanced")
df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
y = df.opportunity_stage_after_30_days.values
plot_learning_curve(estimator,make_scorer(roc_auc_score), X, y, cv=sss, n_jobs=4)
plt.show()
