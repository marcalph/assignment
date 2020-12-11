#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
""" modeling tryouts
"""

import os
import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score 
from sklearn.model_selection import (GridSearchCV, RepeatedKFold,
                                     RepeatedStratifiedKFold,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.preprocessing import LabelEncoder

from utils.log import logger, logthis
from utils.review import *
import click

extra_args = { "funcname_override" : "print"}

# creation 
# status change current/creation + postal code
np.random.seed(42)



useless = ["fake_opportunity_id",       # unique
           "fake_contact_id",           # unique
           "opportunity_stage_after_60_days",
           "opportunity_stage_after_90_days",
           "country"]


dates = ["opportunity_creation_date",   # holidays are visible (lows)
         "contact_creation_date"]       # RFM? start of month highly overrepresented

# arbitrary cutoff @50
cat_hi = ["salesforce_specialty",
          "main_competitor",
          "pms"]#,
        #   "postal_code"]                #4 zeros


cat_lo = ["contact_status",             # attention 8 "obsolète"
          "current_opportunity_stage",
          "opportunity_stage_at_the_time_of_creation",
          "previous_max_stage",
          "pms_status",
          "gender",
          "working_status"]

bools = ["has_mobile_phone",
         "has_website",
         "has_been_recommended",
         "is_city_with_other_clients",
         "is_in_dense_area_for_this_cluster"]

nums = ["count_previous_opportunities",
       "count_total_calls",
       "count_unsuccessful_calls",
       "count_total_appointments",
       "count_contacts_converted_last_30d",
       "count_contacts_converted_last_30d_per_specialty",
       "count_contacts_converted_last_30d_per_zipcode",
       "count_contacts_converted_last_30d_per_specialty_and_zipcode",
       "practitioner_age",
       "years_since_graduation",
       "years_since_last_moving",
       "days_since_last_inbound_lead_date",
       "days_since_last_congress_lead_date",
       "count_clients_with_same_zipcode",
       "count_clients_with_same_zipcode_and_spe",
       "count_clients_with_same_specialty",
       "number_of_prospects_in_account",
       "number_of_clients_in_account"
    ]

target = ["opportunity_stage_after_30_days"
    ]

missings = ["opportunity_stage_after_90_days",
           "previous_max_stage",
           "main_competitor",
           "pms",
           "gender",
           "practioner_age",
           "years_since_graduation",
           "years_since_last_moving",
           "working_status",
           "days_since_last_inbound_lead_date",
           "days_since_last_congress_lead_date",
           "has_been_recommended"]



@logthis
def process_tree(df):
    df["fake_opportunity_id"] = df.fake_opportunity_id.astype(str)
    df["opportunity_creation_date"] = pd.to_datetime(df.opportunity_creation_date)
    df["fake_contact_id"] = df.fake_opportunity_id.astype(str)
    df["contact_creation_date"] = pd.to_datetime(df.contact_creation_date)
    df["postal_code"] = df.postal_code.astype(str)
    df["has_been_recommended"] = df.has_been_recommended.astype(bool)


    # remove useless
    logger.info(df.shape, extra=extra_args)
    df = df.drop(useless, axis=1)
    logger.info(df.shape, extra=extra_args)
    # target
    y = df.opportunity_stage_after_30_days.map({"Missed":0, "Negotiation":0, "Approved":1, "Prospection":0}).values
    df = df.drop("opportunity_stage_after_30_days", axis=1)
    logger.info(df.shape, extra=extra_args)
    # dates
    df["opportunity_year"] = df.opportunity_creation_date.dt.year.astype(str)
    df["opportunity_month"] = df.opportunity_creation_date.dt.month.astype(str)
    df["opportunity_week"] = df.opportunity_creation_date.dt.week.astype(str)
    df["contact_year"] = df.contact_creation_date.dt.year.astype(str)
    df["contact_month"] = df.contact_creation_date.dt.month.astype(str)
    df["contact_week"] = df.contact_creation_date.dt.week.astype(str)
    df = df.drop(dates, axis=1)
    logger.info(df.shape, extra=extra_args)
    # cat
    cat_hi_le = []
    for c in cat_hi:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].fillna("missing"))
        cat_hi_le.append(le)

    cat_lo_le = []
    for c in cat_lo:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].fillna("missing"))
        cat_lo_le.append(le)

    # nums
    for c in nums:
        df[c] = df[c].fillna(-1)

    X = df.values

    return X, y, cat_hi_le+cat_lo_le
    


@logthis
def multi_imputation():
    """ rf mice imputation
    """
    pass


@logthis
def imputation():
    """ simple imputation
    """
    pass





@logthis
def search_model(X, y, shuffle_split_strategy):
    """ replicate described baseline with clean nested CV strategy
        being one of stratified shuffle split, repeated  strat kfold, repeated kfold
    """
    for train_idx, test_idx in shuffle_split_strategy.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        grid = {
            "boosting_type": ["gbdt", "dart"],
            "reg_alpha": np.logspace(0, 2, 4),
            "reg_lambda": np.logspace(0, 2, 4),
            "num_leaves": [int(i) for i in np.linspace(30, 40,3)]
            }
        clf = lgb.LGBMClassifier(objective="binary", class_weight="balanced")
        lr_grid = GridSearchCV(clf, grid, scoring="average_precision", cv=5, n_jobs=3)
        lr_grid.fit(X_train, y_train)
        logger.info(f"LR parameters {lr_grid.best_params_}", extra=extra_args)
        logger.info(f"train avgP :  {lr_grid.best_score_:.4f}, test avgP : {average_precision_score(y_test, lr_grid.best_estimator_.predict(X_test)):.4f}", extra=extra_args)



@logthis
def train_and_serialize(clf, X, y, le_list):
    """ train and serialize simple estimator
    """
    clf.fit(X, y)
    joblib.dump(clf, 'assets/models/lgbm.pkl')
    joblib.dump(le_list, 'assets/models/le_list.pkl')



def process_tree_for_pred(df, le_list):
    """ preprocessing
    """
    df["fake_opportunity_id"] = df.fake_opportunity_id.astype(str)
    df["opportunity_creation_date"] = pd.to_datetime(df.opportunity_creation_date)
    df["fake_contact_id"] = df.fake_opportunity_id.astype(str)
    df["contact_creation_date"] = pd.to_datetime(df.contact_creation_date)
    df["postal_code"] = df.postal_code.astype(str)
    df["has_been_recommended"] = df.has_been_recommended.astype(bool)
    # remove useless
    df = df.drop(useless, axis=1)
    # dates
    df["opportunity_year"] = df.opportunity_creation_date.dt.year.astype(str)
    df["opportunity_month"] = df.opportunity_creation_date.dt.month.astype(str)
    df["opportunity_week"] = df.opportunity_creation_date.dt.week.astype(str)
    df["contact_year"] = df.contact_creation_date.dt.year.astype(str)
    df["contact_month"] = df.contact_creation_date.dt.month.astype(str)
    df["contact_week"] = df.contact_creation_date.dt.week.astype(str)
    df = df.drop(dates, axis=1)
    for c, le in zip(cat_hi+cat_lo, le_list):
        df[c] = le.transform(df[c].fillna("missing"))
    
    for c in nums:
        df[c] = df[c].fillna(-1)

    X = df.values
    feat_names = df.columns.values

    return X, feat_names



class CatDataset(Dataset):
    """ torch dataset for pandas dataframe :)
        embedding_cols = {colname: len(col.cat.categories)}
    """
    def __init__(self, X, y, embedding_cols):
        X = X.copy()
        self.Xcat = X.loc[:, embedding_cols.keys()].copy().values.astype(np.int64)
        self.Xnum = X.drop(columns=embedding_cols.keys()).copy().values.astype(np.float32)
        self.y = y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.Xcat[idx], self.Xnum[idx], self.Y[idx]




class SimpleNN(nn.Module):
    """ simple FFNN for tabular data
        emedding_sizes = [(n_categories, min(30 to 50, (n_categories+1)//2))]
    """
    def __init__(self, embedding_sizes, n_num):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_num = n_emb, n_num
        self.lin1 = nn.Linear(self.n_emb + self.n_num, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(self.n_num)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.embedding_dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x_cat, x_num):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.embedding_dropout(x)
        xnum = self.bn1(x_num)
        x = torch.cat([x, xnum], 1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x



from utils.log import logger, logthis
extra_args = { "funcname_override" : "print"}

@click.command()
@click.option("--search", is_flag=True)
@click.option("--diagnose", is_flag=True)
@click.option("--train", is_flag=True)
def main(**kwargs):
    df = pd.read_csv("assets/subject/case_study_scoring_raw.csv")
    sss = StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=42)
    X, y, le_list = process_tree(df) 
    clf = lgb.LGBMClassifier(objective="binary", class_weight="balanced", boosting_type="gbdt", num_leaves=40, reg_alpha=1, reg_lambda=1)
    if kwargs["search"]:
        search_baseline(df, sss)
    if kwargs["diagnose"]:
        plot_learning_curves(clf,make_scorer(roc_auc_score), X, y, filename="assets/output/figs/LC_lgbm_test.png", cv=sss, n_jobs=3)
        cv_confusion_matrix(clf, X, y, sss, filename="assets/output/figs/CM_lgbm_test.png", normalize=False)
        cv_confusion_matrix(clf, X, y, sss, filename="assets/output/figs/CMprec_lgbm_test.png", normalize=True)
    if kwargs["train"]:
        train_and_serialize(clf, X, y, le_list)



if __name__ == "__main__":
    main()


