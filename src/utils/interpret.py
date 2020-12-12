#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" interpretabilty utils
"""
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def interpret_global_shap_linear(clf, X, y):
    """generate shap dot plot for a linear model
    todo CV
    """
    explainer = shap.LinearExplainer(clf, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)  # should be test
    cmap = sns.diverging_palette(250, 30, as_cmap=True)
    shap.summary_plot(
        shap_values,
        X,
        cmap=cmap,
        show=False,
        max_display=10,
        plot_size=(5, 5),
        plot_type="dot",
        feature_names=range(X.shape[1]),
    )
    fig = plt.gcf()
    plt.xlabel("SHAP")
    ax_list = fig.axes
    ax_list[1].remove()
    plt.savefig("/volume/shap_imptce.html")


def interpret_sample_shap_linear(clf, X, y, sample):
    """additive force plot, todo handle spe
    todo CV
    """
    explainer = shap.LinearExplainer(clf, X, feature_perturbation="independent")
    shap_values = explainer.shap_values(X)  # should be test
    shap.force_plot(
        explainer.expected_value,
        shap_values[sample.name],
        sample,
        matplotlib=True,
        show=False,
    )
    plt.savefig("/volume/shap_result.html")


def interpret_sample_lime(clf, X, y, sample, names):
    """compute lime explainer and generate"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.astype(float).astype(int),
        mode="classification",
        training_labels=y,
        feature_names=names,
        class_names=[0, 1],
    )
    limexpl = explainer.explain_instance(
        sample.astype(float).astype(int).flatten(),
        clf.predict_proba,
        num_features=10,
        num_samples=500,
    )
    limexpl.save_to_file("/volume/lime_result_lgbm.html")
    return limexpl.as_html()
