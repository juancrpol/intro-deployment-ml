import logging
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from utils import (
    get_model_performance_test_set,
    save_simple_metrics_report,
    update_model,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

logger.info("Loading Data...")
data = pd.read_csv("dataset/full_data.csv")

logger.info("Loading Model...")
model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
        ("core_model", GradientBoostingRegressor()),
    ]
)

logger.info("Separating dataset info train and test")
X = data.drop(["worldwide_gross"], axis=1)
y = data["worldwide_gross"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42
)

logger.info("Setting Hyperparameters to tune")
param_tuning = {"core_model__n_estimators": range(20, 301, 20)}

grid_search = GridSearchCV(model, param_grid=param_tuning, scoring="r2", cv=5)

logger.info("Starting grid search...")
grid_search.fit(X_train, y_train)

logger.info("Cross Validating with best model...")
final_result = cross_validate(
    grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5
)

train_score = np.mean(final_result["train_score"])
test_score = np.mean(final_result["test_score"])
assert train_score > 0.7
assert test_score > 0.65

logger.info(f"Train Score: {train_score}")
logger.info(f"Test Score: {test_score}")

logger.info("Updating model...")
update_model(grid_search.best_estimator_)

logger.info("Generating model report...")
validation_score = grid_search.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(
    train_score, test_score, validation_score, grid_search.best_estimator_
)

y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

logger.info("Training finished")
