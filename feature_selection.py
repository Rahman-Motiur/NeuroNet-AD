
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def ensemble_feature_selection(df_meta: pd.DataFrame,
                               target: pd.Series,
                               k: int = 5,
                               random_state: int = 42) -> Tuple[List[str], pd.DataFrame]:

    features = df_meta.columns.tolist()
    X = df_meta.values
    y = target.values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    models = {
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "xgb": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, random_state=random_state),
        "lgbm": LGBMClassifier(n_estimators=500, max_depth=-1, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, random_state=random_state),
        "extra": ExtraTreesClassifier(n_estimators=400, random_state=random_state),
        "ada": AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state),
    }

    votes = np.zeros(len(features), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for name, clf in models.items():
        fold_imps = np.zeros(len(features))
        for tr, va in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
            else:
                imp = np.zeros(len(features))
            fold_imps += imp
        fold_imps /= skf.get_n_splits()
        order = np.argsort(-fold_imps)  # descending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(features) + 1)
        votes += 1.0 / ranks

    scores = votes / votes.sum()
    ranking = pd.DataFrame({"feature": features, "score": scores}).sort_values("score", ascending=False)
    top_features = ranking.head(k)["feature"].tolist()
    return top_features, ranking
