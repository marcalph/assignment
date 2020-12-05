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

    # return plt




cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = LogisticRegression(C=100, class_weight="balanced")
df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
y = df.opportunity_stage_after_30_days.values
plot_learning_curve(estimator,make_scorer(roc_auc_score), X, y, cv=cv, n_jobs=4)
plt.show()



def create_confusion_matrix(model="cnn_spacy"):
    '''
    Script to create a confusion matrix
    '''
    dataframe = pd.read_pickle(f'/app/assets/data/test/predicted_data_{model}.pkl')

    dataframe[model] = dataframe[model].apply(convert_prediction_to_label, model=model)

    fig = plt.figure(figsize=(38.40, 21.60), dpi=100)
    y_true = np.array([np.array(x) for x in dataframe['label']]).astype(np.int8)
    y_pred = np.array([np.array(x) for x in dataframe[model]]).astype(np.int8)

    print(y_pred)
    print(type(y_pred))
    print(y_pred.dtype)

    class_names = LIST_INTENTION_FINAL

    conf_mat = multilabel_confusion_matrix(y_true, y_pred, labels=class_names).T

    df_cm = pd.DataFrame(
            conf_mat/conf_mat.sum(axis=1)[:, np.newaxis], index=class_names, columns=class_names
        )
    print(df_cm)

    heatmap = sns.heatmap(df_cm, annot=True, fmt=".1%", annot_kws={"size": 10}, cmap="coolwarm")#sns.light_palette((210, 90, 60), input="husl"))

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=6)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(f'app/stats/confmat_norm_{model}.png')


def validate(X_test, y_test, nlp):
    logger.info("start validation")
    df = pd.DataFrame()
    df["text"] = X_test
    df["target"] = y_test
    df["pred_label"] = X_test.apply(lambda x: nlp(x).cats)
    df["pred_target"] = df.pred_label.apply(encode_pred_label)
    logger.warn(df.head())
    logger.info("prediction on test data done")
    y_test = np.vstack(df.target)
    y_pred = np.vstack(df.pred_target)
    logger.info(y_test.shape)
    logger.info(y_pred.shape)
    logger.info(INTENTS)
    class_names = INTENTS
    logger.info("start classification report generation")
    return classification_report(y_test, y_pred, target_names=class_names)









fig, axes = plt.subplots(1, 3, figsize=(30, 5))


cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = LogisticRegression(C=100, class_weight="balanced")
df = pd.read_csv("assets/assignment/case_study_scoring_clean.csv", sep=";")
X = df.drop(["opportunity_stage_after_30_days"],axis=1).values
y = df.opportunity_stage_after_30_days.values
plot_learning_curve(estimator,make_scorer(roc_auc_score), X, y, axes=axes, cv=cv, n_jobs=4)

plt.show()






