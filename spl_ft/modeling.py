"""Model definitions and evaluation helpers.

Simple defaults only, per the assignment ("do not perform unnecessary
hyperparameter tuning"). All imputation/scaling lives inside sklearn
Pipelines so it is only ever fit on training data.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_COLUMNS


def build_models(seed: int = config.RANDOM_SEED) -> Dict[str, Pipeline]:
    """Returns {name: unfitted sklearn Pipeline}."""
    majority = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", DummyClassifier(strategy="most_frequent", random_state=seed)),
    ])

    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])

    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=seed)),
    ])

    return {
        "majority_baseline": majority,
        "logistic_regression": logreg,
        "random_forest": rf,
    }


def load_usable_features(path=config.FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    usable = df[~df["excluded"]].copy()
    return usable.reset_index(drop=True)


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y
