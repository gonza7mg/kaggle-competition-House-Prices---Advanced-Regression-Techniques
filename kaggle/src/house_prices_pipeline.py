from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "SalePrice"
ID_COLUMN = "Id"

NONE_COLUMNS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ZERO_COLUMNS = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]

CATEGORICAL_NUMERIC_COLUMNS = ["MSSubClass", "MoSold", "YrSold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a House Prices model and generate a Kaggle submission."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmsle(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return float(
        np.sqrt(mean_squared_error(np.log1p(y_true_array), np.log1p(y_pred_array)))
    )


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    missing = [path for path in [train_path, test_path, sample_path] if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"No encuentro estos archivos: {missing_names}. "
            "Descargalos con scripts/download_data.ps1 o colocalos en data/raw."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    return train, test, sample_submission


def clean_known_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in NONE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None")

    for column in ZERO_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    for column in ["Functional", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]:
        if column in df.columns:
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda values: values.fillna(values.median())
        )

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in CATEGORICAL_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(str)

    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
    )
    df["TotalBathrooms"] = (
        df.get("FullBath", 0)
        + 0.5 * df.get("HalfBath", 0)
        + df.get("BsmtFullBath", 0)
        + 0.5 * df.get("BsmtHalfBath", 0)
    )
    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0)
        + df.get("EnclosedPorch", 0)
        + df.get("3SsnPorch", 0)
        + df.get("ScreenPorch", 0)
    )
    df["HouseAge"] = df.get("YrSold", 0).astype(int) - df.get("YearBuilt", 0)
    df["RemodAge"] = df.get("YrSold", 0).astype(int) - df.get("YearRemodAdd", 0)
    df["GarageAge"] = df.get("YrSold", 0).astype(int) - df.get("GarageYrBlt", 0)
    df["GarageAge"] = df["GarageAge"].clip(lower=0)

    df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int)
    df["HasGarage"] = (df.get("GarageArea", 0) > 0).astype(int)
    df["HasBasement"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
    df["Has2ndFloor"] = (df.get("2ndFlrSF", 0) > 0).astype(int)
    df["IsRemodeled"] = (df.get("YearRemodAdd", 0) != df.get("YearBuilt", 0)).astype(int)

    if "OverallQual" in df.columns:
        df["OverallQualSquared"] = df["OverallQual"] ** 2

    return df


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = clean_known_missing_values(train)
    test = clean_known_missing_values(test)

    y = train[TARGET].copy()
    train_features = train.drop(columns=[TARGET])
    combined = pd.concat([train_features, test], axis=0, ignore_index=True)
    combined = add_features(combined)

    train_processed = combined.iloc[: len(train_features)].copy()
    test_processed = combined.iloc[len(train_features) :].copy()

    for frame in [train_processed, test_processed]:
        if ID_COLUMN in frame.columns:
            frame.drop(columns=[ID_COLUMN], inplace=True)

    return train_processed, y, test_processed


def get_column_groups(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return numeric_columns, categorical_columns


def make_preprocessor(X: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns, categorical_columns = get_column_groups(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_columns),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(X: pd.DataFrame, seed: int) -> Dict[str, Pipeline]:
    linear_preprocessor = make_preprocessor(X, scale_numeric=True)
    tree_preprocessor = make_preprocessor(X, scale_numeric=False)

    return {
        "ridge": Pipeline(
            [
                ("preprocess", clone(linear_preprocessor)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=Ridge(alpha=12.0),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "lasso": Pipeline(
            [
                ("preprocess", clone(linear_preprocessor)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=Lasso(alpha=0.0005, max_iter=50000, random_state=seed),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "elastic_net": Pipeline(
            [
                ("preprocess", clone(linear_preprocessor)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=ElasticNet(
                            alpha=0.0007,
                            l1_ratio=0.8,
                            max_iter=50000,
                            random_state=seed,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", clone(tree_preprocessor)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=GradientBoostingRegressor(
                            n_estimators=1200,
                            learning_rate=0.025,
                            max_depth=3,
                            min_samples_leaf=3,
                            min_samples_split=10,
                            subsample=0.75,
                            random_state=seed,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
    }


def cross_validate_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for model_name, model in models.items():
        fold_scores = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            estimator = clone(model)
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_valid)
            score = rmsle(y_valid, predictions)
            fold_scores.append(score)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "rmsle": score,
                }
            )

        print(f"{model_name}: RMSLE {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("model")["rmsle"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("mean")
    )
    print("\nResumen CV:")
    print(summary.to_string(index=False))
    return results


def make_ensemble_predictions(
    models: Dict[str, Pipeline],
    cv_results: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    cv_summary = cv_results.groupby("model")["rmsle"].mean().sort_values()
    selected_models = cv_summary.head(3)
    inverse_scores = 1 / selected_models
    weights = inverse_scores / inverse_scores.sum()

    print("\nModelos usados en el ensamble:")
    for model_name, weight in weights.items():
        print(f"- {model_name}: peso {weight:.3f}")

    ensemble_predictions = np.zeros(len(X_test))
    for model_name, weight in weights.items():
        estimator = clone(models[model_name])
        estimator.fit(X, y)
        ensemble_predictions += weight * estimator.predict(X_test)

    return pd.Series(np.maximum(ensemble_predictions, 0), name=TARGET)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train, test, sample_submission = load_data(args.data_dir)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    X, y, X_test = prepare_features(train, test)

    print(f"Train: {X.shape[0]} filas, {X.shape[1]} variables")
    print(f"Test:  {X_test.shape[0]} filas, {X_test.shape[1]} variables")

    models = build_models(X, args.seed)
    cv_results = cross_validate_models(
        models,
        X,
        y,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    cv_output = args.output_dir / "cv_results.csv"
    cv_results.to_csv(cv_output, index=False)

    predictions = make_ensemble_predictions(models, cv_results, X, y, X_test)
    submission = sample_submission.copy()
    submission[TARGET] = predictions

    submission_output = args.output_dir / "submission.csv"
    submission.to_csv(submission_output, index=False)

    print(f"\nResultados guardados en: {cv_output}")
    print(f"Submission guardada en: {submission_output}")


if __name__ == "__main__":
    main()
