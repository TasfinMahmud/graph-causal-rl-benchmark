import numpy as np
from econml.dml import CausalForestDML
from econml.dr import ForestDRLearner
from econml.metalearners import XLearner
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

def get_grf_model():
    """
    Generalized Random Forests (GRF) implemented via CausalForestDML.
    Uses Random Forests for the outcome and treatment models to maintain stability.
    """
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1),
        discrete_treatment=True,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    return model

def get_forestdr_model():
    """
    Forest Doubly Robust Learner (deprecated shortcut).
    """
    return ForestDRLearner(
        model_regression=RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
        model_propensity=RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1),
        n_crossfit_splits=2,
        min_propensity=1e-6
    )

def get_xlearner_model():
    """
    Exact P8 Baseline: Meta-learner (X-Learner).
    """
    return XLearner(
        models=LGBMRegressor(n_estimators=100, max_depth=10),
        propensity_model=LogisticRegression(max_iter=1000)
    )

def get_slearner_model():
    from econml.metalearners import SLearner
    return SLearner(
        overall_model=LGBMRegressor(n_estimators=100, max_depth=10)
    )
