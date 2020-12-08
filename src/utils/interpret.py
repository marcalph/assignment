#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" interpretabilty utils
"""
import shap
import matplotlib.pyplot as plt
import seaborn as sns

import lime
import lime.lime_tabular


from utils.log import logger, logthis
extra_args = { "funcname_override" : "print"}



@logthis
def interpret_global_shap_linear(clf, X, y):
    """ generate shap dot plot for a linear model
        todo CV
    """
    explainer = shap.LinearExplainer(clf, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X) # should be test
    cmap = sns.diverging_palette(250, 30, as_cmap=True)
    shap.summary_plot(shap_values, X, cmap=cmap, show=False, max_display=10, plot_size=(5,5), plot_type="dot", feature_names=range(X.shape[1]))
    fig = plt.gcf()
    plt.xlabel("SHAP")
    ax_list=fig.axes           
    ax_list[1].remove()
    plt.savefig("/volume/shap_imptce.html")



@logthis
def interpret_sample_shap_linear(clf, X, y, sample):
    """ additive force plot, todo handle spe
        todo CV
    """
    explainer = shap.LinearExplainer(clf, X, feature_perturbation="independent")
    shap_values = explainer.shap_values(X)# should be test
    shap.force_plot(explainer.expected_value, shap_values[sample.name], sample, matplotlib=True, show=False)
    fig = plt.gcf()
    plt.savefig("/volume/shap_result.html")



@logthis
def interpret_sample_lime(clf, X, y, sample):
    """ compute lime explainer and generate 
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(X, 
                        mode="classification",
                        training_labels=y,
                        feature_names=sample.index.values,
                        class_names=[0,1])
    limexpl = explainer.explain_instance(sample, clf.predict_proba, num_features=5)
    limexpl.save_to_file("/volume/lime_result.html")





