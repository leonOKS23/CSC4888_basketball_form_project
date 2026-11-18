"""Train + evaluate make/miss classifiers on SPL features.

IN:   data/processed/spl_features.csv  (from SPL_to_features.py)
OUT:  printed metrics (accuracy, balanced accuracy, F1 for 'made', confusion matrix)

Models: majority-class baseline, logistic regression, random forest. Hyperparameters
are chosen by GridSearchCV on the TRAINING rows only, with folds grouped by player
(so the search rewards generalizing across people, not memorizing one shooter).

Evaluation A: known players, later shots - per player, first 80% of trials (by
              trial number) train, last 20% test.
Evaluation B: unseen players - leave one player out, refit everything per fold.

Uses the 2025-12-18 session only (5 players, 60 FPS); see DATA_NOTES.md.
Run:  .venv/bin/python scripts/train_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pose_features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
SEED = 0


def load(session='2025-12-18'):
    df = pd.read_csv(ROOT / 'data/processed/spl_features.csv')
    return df[df.session == session].sort_values(['participant_id', 'trial_num']).reset_index(drop=True)


def models():
    """name -> (pipeline, param grid). Scaling lives inside the pipeline (fit on train only)."""
    return {
        'baseline': (Pipeline([('clf', DummyClassifier(strategy='most_frequent'))]), {}),
        'logreg': (
            Pipeline([('scale', StandardScaler()),
                      ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))]),
            {'clf__C': [0.01, 0.1, 1, 10]},
        ),
        'forest': (
            Pipeline([('clf', RandomForestClassifier(class_weight='balanced', n_estimators=300,
                                                     random_state=SEED, n_jobs=-1))]),
            {'clf__max_depth': [2, 4, 8, None], 'clf__min_samples_leaf': [1, 5, 15]},
        ),
    }


def fit(name, train):
    pipe, grid = models()[name]
    X, y, groups = train[FEATURES], train.label, train.participant_id
    if not grid:
        return pipe.fit(X, y), {}
    search = GridSearchCV(pipe, grid, scoring='balanced_accuracy',
                          cv=GroupKFold(n_splits=groups.nunique()))
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def report(name, y, pred, extra=''):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f'  {name:9s} acc {accuracy_score(y, pred):.3f}  bal_acc {balanced_accuracy_score(y, pred):.3f}  '
          f'F1(made) {f1_score(y, pred, zero_division=0):.3f}  '
          f'CM [[TN {tn} FP {fp}] [FN {fn} TP {tp}]]  {extra}')


def eval_a(df):
    rank = df.groupby('participant_id').cumcount()
    size = df.groupby('participant_id').trial_num.transform('size')
    is_test = rank >= np.ceil(0.8 * size)
    train, test = df[~is_test], df[is_test]
    print(f'\n=== Eval A: known players, later shots  (train {len(train)}, test {len(test)}, '
          f'test make rate {test.label.mean():.2f}) ===')
    for name in models():
        model, params = fit(name, train)
        report(name, test.label, model.predict(test[FEATURES]), params or '')


def eval_b(df):
    print(f'\n=== Eval B: leave-one-player-out  (n {len(df)}, make rate {df.label.mean():.2f}) ===')
    for name in models():
        preds = pd.Series(index=df.index, dtype=float)
        for player in df.participant_id.unique():
            held = df.participant_id == player
            model, _ = fit(name, df[~held])
            preds[held] = model.predict(df.loc[held, FEATURES])
        report(name, df.label, preds.astype(int), '(pooled)')
        for player in df.participant_id.unique():
            held = df.participant_id == player
            print(f'      {player}: n {held.sum():3d}  make rate {df.label[held].mean():.2f}  '
                  f'acc {accuracy_score(df.label[held], preds[held]):.3f}  '
                  f'bal_acc {balanced_accuracy_score(df.label[held], preds[held]):.3f}')


if __name__ == '__main__':
    df = load()
    print(f'{len(df)} shots, {df.participant_id.nunique()} players, features: {FEATURES}')
    eval_a(df)
    eval_b(df)
