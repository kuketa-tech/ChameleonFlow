from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SensorModelSpec:
    name: str
    serializer: str
    builder: Callable[[int], Any]


def _build_lightgbm(seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
    )


def _build_lightgbm_sigmoid(seed: int) -> Any:
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV

    estimator = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
    )
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)


def _build_hist_gradient_boosting(seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed,
    )


def _build_random_forest(seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def _build_extra_trees(seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def _build_logistic_regression(seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


MODEL_SPECS: dict[str, SensorModelSpec] = {
    "lightgbm": SensorModelSpec(
        name="lightgbm",
        serializer="lightgbm_booster",
        builder=_build_lightgbm,
    ),
    "lightgbm_sigmoid": SensorModelSpec(
        name="lightgbm_sigmoid",
        serializer="joblib",
        builder=_build_lightgbm_sigmoid,
    ),
    "hist_gradient_boosting": SensorModelSpec(
        name="hist_gradient_boosting",
        serializer="joblib",
        builder=_build_hist_gradient_boosting,
    ),
    "random_forest": SensorModelSpec(
        name="random_forest",
        serializer="joblib",
        builder=_build_random_forest,
    ),
    "extra_trees": SensorModelSpec(
        name="extra_trees",
        serializer="joblib",
        builder=_build_extra_trees,
    ),
    "logistic_regression": SensorModelSpec(
        name="logistic_regression",
        serializer="joblib",
        builder=_build_logistic_regression,
    ),
}


def available_sensor_algorithms() -> list[str]:
    return list(MODEL_SPECS)


def build_sensor_estimator(algorithm: str, *, seed: int) -> Any:
    try:
        spec = MODEL_SPECS[algorithm]
    except KeyError as exc:
        msg = f"Unknown sensor algorithm: {algorithm}"
        raise ValueError(msg) from exc

    return spec.builder(seed)


def get_sensor_model_spec(algorithm: str) -> SensorModelSpec:
    try:
        return MODEL_SPECS[algorithm]
    except KeyError as exc:
        msg = f"Unknown sensor algorithm: {algorithm}"
        raise ValueError(msg) from exc
