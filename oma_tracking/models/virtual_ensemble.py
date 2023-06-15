"""Virtual Ensemble models to quantify uncertainty in predictions.
"""

import numpy as np
from xgboost import XGBRegressor
import pandas as pd

class VirtualEnsembleXGB:
    def __init__(self, xgb_model: XGBRegressor, n_submodels: int):
        self.xgb_model = xgb_model
        self.n_submodels = n_submodels

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        booster = self.xgb_model.get_booster()
        total_trees = booster.best_iteration

        for _ in range(self.n_submodels):
            n_trees = np.random.randint(total_trees/2, total_trees)
            predictions.append(self.xgb_model.predict(X, iteration_range=(0,n_trees)))
        
        return np.array(predictions)
    
    def mean_predictions(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        return np.mean(predictions, axis=0)

    def std_predictions(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        return np.std(predictions, axis=0)
