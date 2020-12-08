#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" modeling tryouts
"""

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, train_test_split, StratifiedShuffleSplit, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import click
import joblib

from utils.review import *

from utils.log import logger, logthis
extra_args = { "funcname_override" : "print"}

np.random.seed(42)






