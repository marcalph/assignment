#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" training  diagnostics utils
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix

from log import logger, logthis
extra_args = { "funcname_override" : "print"}


@logthis
def plot_learning_curve(clf, scoring, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """ Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf, X, y, cv=cv, n_jobs=n_jobs,
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
    plt.show()


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
clf = LogisticRegression(C=100, class_weight="balanced")
df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
y = df.opportunity_stage_after_30_days.values
plot_learning_curve(clf,make_scorer(roc_auc_score), X, y, cv=sss, n_jobs=4)





@logthis
def most_incorrect(clf, X, y, shuffle_split_strategy):
    """ compute most incorrect predictions given clf and CV strategy
        returns list of incorrect predictions, sorted by decreasing model confidence
        NB: this should ideally be done w/ trained estimator in a formal test setting
    """
    incorrect = []
    for train_idx, test_idx in shuffle_split_strategy.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:,1]
        # aggregate wrong predictions
        wrong = [(x, yp, y, p) for x, yp, y, p in list(zip(X_test, preds, y_test, probas)) if yp!=y]
        # filter to get wrong predictions sorted by 
        wrong.sort(key= lambda tup: abs(tup[-1]-.5))
        incorrect.append(wrong)
    return incorrect



@logthis
def generate_incorrect_report(incorrect_list, columns):
    """ given list of incorrect predictions generate profile report
    """
    # make incorrect df to find patterns
    data = np.vstack([np.hstack([tup[2], tup[0]]) for list_ in incorrect_list for tup in list_ ])
    incorrect_df = pd.DataFrame(data, columns=columns)
    profile = ProfileReport(incorrect_df, explorative=True)
    profile.to_file("assets/wip/report_incorrect.html")


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
clf = LogisticRegression(C=100, class_weight="balanced")
df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
y = df.opportunity_stage_after_30_days.values
incorrect = most_incorrect(clf, X, y, sss)
generate_incorrect_report(incorrect, df.columns)



@logthis
def cv_confusion_matrix(clf, X, y, shuffle_split_strategy, normalize=False):
    """ compute a cross validated confusion matrix for a bin target
    """
    cms = []
    labels = [1, 0]
    for train_idx, test_idx in shuffle_split_strategy.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels).T
        cms.append(cm)
    
    agg_cm = np.mean(cms, axis=0)
    if normalize:
        cm_df = pd.DataFrame(
            agg_cm/agg_cm.sum(axis=1)[:, np.newaxis], index=labels, columns=labels)
        hm = sns.heatmap(cm_df, linewidth=.5, annot=True, fmt=".1%", annot_kws={"size": 10}, cbar=False)
    else:
        cm_df = pd.DataFrame(
            agg_cm, index=labels, columns=labels) 
        hm = sns.heatmap(cm_df, linewidth=.5, annot=True, fmt=".0f", annot_kws={"size": 10}, cbar=False)
    
    hm.set_xlabel("True")
    hm.set_ylabel("Pred")
    plt.show()    


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
clf = LogisticRegression(C=100, class_weight="balanced")
df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
y = df.opportunity_stage_after_30_days.values
cv_confusion_matrix(clf, X, y, sss)



@logthis
def cv_classification_report():
    """ generate a cross validated classification report
    """    
    pass

