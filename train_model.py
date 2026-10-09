"""
train_model.py
--------------
Step 2: Data Cleaning, Feature Engineering, 5-Fold Cross-Validation,
        Hyperparameter Tuning (GridSearchCV) & Artifact Export

Upgraded Version:
- 5-Fold Stratified Cross-Validation on all candidate models.
- Light hyperparameter tuning with GridSearchCV on the best model.
- Generates ROC curve coordinates and Confusion Matrix data for interactive charts.
- Maps customer purchase histories to support excluding previously purchased items.
- Strict Zero Data Leakage: 'purchases' is strictly excluded from feature matrix X.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

def load_and_clean_data(filepath: str = "data.csv"):
    """
    Loads raw CSV and performs data cleaning:
    - Removes duplicate rows
    - Imputes numeric columns with their median
    - Imputes categorical columns with their mode
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filepath}' not found! Run generate_data.py first.")

    raw_df = pd.read_csv(filepath)
    initial_rows = len(raw_df)

    # 1. Deduplicate
    duplicates_count = int(raw_df.duplicated().sum())
    df = raw_df.drop_duplicates().reset_index(drop=True)
    post_dedup_rows = len(df)

    missing_before = raw_df.isnull().sum().to_dict()

    # 2. Impute numeric columns with median
    num_cols = ["product_price", "views", "cart_adds", "previous_purchases", "purchases"]
    imputed_values = {}
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_values[col] = f"Median: {median_val:.2f}"

    # 3. Impute categorical columns with mode
    cat_cols = ["product_category"]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            imputed_values[col] = f"Mode: {mode_val}"

    # Logical integrity check
    df["cart_adds"] = df["cart_adds"].clip(upper=df["views"])

    cleaning_stats = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_count,
        "clean_rows": post_dedup_rows,
        "missing_before": missing_before,
        "imputed_strategy": imputed_values
    }

    return df, cleaning_stats


def create_price_buckets(price_series: pd.Series) -> pd.Series:
    """Categorizes prices into 4 intuitive tiers."""
    bins = [-np.inf, 50, 150, 400, np.inf]
    labels = ["Budget", "Mid-Range", "Premium", "Luxury"]
    return pd.cut(price_series, bins=bins, labels=labels)


def engineer_features(df: pd.DataFrame):
    """
    Transforms clean raw columns into predictive machine learning features.
    STRICT ZERO-LEAKAGE: 'purchases' is ONLY used for ground-truth label 'will_purchase'.
    """
    data = df.copy()

    # Ground-truth binary target
    y = (data["purchases"] > 0).astype(int)

    # Feature 1: Cart rate
    data["cart_rate"] = np.where(data["views"] > 0, data["cart_adds"] / data["views"], 0.0)

    # Feature 2: Intent ratio
    data["cart_intent_ratio"] = (data["cart_adds"] * 2.0) / (data["views"] + 1.0)

    # Feature 3: Customer loyalty score
    data["loyalty_score"] = data["previous_purchases"] / (data["previous_purchases"] + 5.0)

    # Feature 4: Price buckets
    data["price_bucket"] = create_price_buckets(data["product_price"])

    # Feature 5: Price per view
    data["price_per_view"] = data["product_price"] / (data["views"] + 1.0)

    # Features to keep in X (Strictly no customer_id, product_id, or purchases)
    feature_columns = [
        "product_category",
        "product_price",
        "views",
        "cart_adds",
        "previous_purchases",
        "cart_rate",
        "cart_intent_ratio",
        "loyalty_score",
        "price_bucket",
        "price_per_view"
    ]

    X = data[feature_columns]
    return X, y, data


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series):
    """
    Performs:
    1. Stratified 80/20 train/test split.
    2. 5-Fold Stratified Cross-Validation on base models.
    3. Hyperparameter tuning via GridSearchCV on the best model.
    4. Computes ROC curves and Confusion Matrix.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [
        "product_price", "views", "cart_adds",
        "previous_purchases", "cart_rate", "cart_intent_ratio",
        "loyalty_score", "price_per_view"
    ]
    categorical_features = ["product_category", "price_bucket"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    # 1. 5-Fold Stratified Cross-Validation
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_summary = []
    trained_pipelines = {}
    test_metrics = []
    roc_curves = {}

    for name, model in base_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Run 5-fold CV on training data
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv_kfold, scoring=scoring)
        cv_summary.append({
            "Model": name,
            "CV Accuracy (Mean +/- Std)": f"{cv_scores['test_accuracy'].mean():.4f} +/- {cv_scores['test_accuracy'].std():.4f}",
            "CV F1 (Mean +/- Std)": f"{cv_scores['test_f1'].mean():.4f} +/- {cv_scores['test_f1'].std():.4f}",
            "CV ROC-AUC (Mean +/- Std)": f"{cv_scores['test_roc_auc'].mean():.4f} +/- {cv_scores['test_roc_auc'].std():.4f}",
            "mean_roc_auc": cv_scores["test_roc_auc"].mean()
        })

        # Fit on full training split and evaluate on test split
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        test_metrics.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(auc, 4)
        }

    cv_results_df = pd.DataFrame(cv_summary)
    test_metrics_df = pd.DataFrame(test_metrics)

    # 2. Hyperparameter Tuning on Best Base Model
    best_base_name = cv_results_df.sort_values(by="mean_roc_auc", ascending=False).iloc[0]["Model"]
    print(f"\nTop Base Model from Cross-Validation: {best_base_name}")

    if best_base_name == "Logistic Regression":
        tune_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42))
        ])
        param_grid = {
            "classifier__C": [0.1, 1.0, 5.0],
            "classifier__max_iter": [500, 1000]
        }
    elif best_base_name == "Random Forest":
        tune_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])
        param_grid = {
            "classifier__n_estimators": [80, 120],
            "classifier__max_depth": [6, 10, None],
            "classifier__min_samples_split": [2, 5]
        }
    else:  # Gradient Boosting
        tune_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=42))
        ])
        param_grid = {
            "classifier__n_estimators": [80, 120],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }

    print("Running GridSearchCV for hyperparameter optimization (5 folds)...")
    grid_search = GridSearchCV(
        tune_pipeline,
        param_grid=param_grid,
        cv=cv_kfold,
        scoring="roc_auc",
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params_clean = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    best_cv_score = grid_search.best_score_
    print(f"Optimal Hyperparameters: {best_params_clean}")
    print(f"Best CV ROC-AUC: {best_cv_score:.4f}")

    # Evaluate Tuned Model on Test Set
    y_pred_tuned = best_pipeline.predict(X_test)
    y_proba_tuned = best_pipeline.predict_proba(X_test)[:, 1]

    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_prec = precision_score(y_test, y_pred_tuned, zero_division=0)
    tuned_rec = recall_score(y_test, y_pred_tuned, zero_division=0)
    tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
    tuned_auc = roc_auc_score(y_test, y_proba_tuned)

    tuned_model_label = f"Tuned {best_base_name} (Optimized)"
    test_metrics_df = pd.concat([
        test_metrics_df,
        pd.DataFrame([{
            "Model": tuned_model_label,
            "Accuracy": round(tuned_acc, 4),
            "Precision": round(tuned_prec, 4),
            "Recall": round(tuned_rec, 4),
            "F1 Score": round(tuned_f1, 4),
            "ROC-AUC": round(tuned_auc, 4)
        }])
    ], ignore_index=True)

    # ROC Curve for Tuned Model
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_proba_tuned)
    roc_curves[tuned_model_label] = {
        "fpr": fpr_tuned.tolist(),
        "tpr": tpr_tuned.tolist(),
        "auc": round(tuned_auc, 4)
    }

    # Confusion Matrix for Tuned Model
    cm = confusion_matrix(y_test, y_pred_tuned)
    confusion_matrix_data = {
        "matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "total_test": len(y_test)
    }

    # 3. Extract Feature Importances from Tuned Model
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor_fitted.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names

    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        raw_importances = np.abs(classifier.coef_[0])
    else:
        raw_importances = np.ones(len(all_feature_names))

    feature_importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": raw_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "best_model_name": tuned_model_label,
        "best_pipeline": best_pipeline,
        "best_params": best_params_clean,
        "cv_results_df": cv_results_df.drop(columns=["mean_roc_auc"]),
        "test_metrics_df": test_metrics_df,
        "roc_curves": roc_curves,
        "confusion_matrix_data": confusion_matrix_data,
        "feature_importance_df": feature_importance_df
    }


def run_pipeline():
    print("=" * 60)
    print("STEP 1: Cleaning Synthetic Dataset")
    print("=" * 60)
    df_clean, cleaning_stats = load_and_clean_data("data.csv")
    print(f"Initial Rows: {cleaning_stats['initial_rows']}")
    print(f"Duplicates Dropped: {cleaning_stats['duplicates_removed']}")
    print(f"Clean Records: {cleaning_stats['clean_rows']}")

    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering (Zero Leakage)")
    print("=" * 60)
    X, y, df_enriched = engineer_features(df_clean)
    print(f"Features: {list(X.columns)}")
    print(f"Conversion Rate: {y.mean() * 100:.2f}%")

    print("\n" + "=" * 60)
    print("STEP 3: 5-Fold Cross-Validation & Light Hyperparameter Tuning")
    print("=" * 60)
    results = train_and_evaluate_models(X, y)

    print("\n5-Fold Stratified Cross-Validation Summary:")
    print(results["cv_results_df"].to_string(index=False))

    print("\nHoldout Test Set Benchmark Leaderboard:")
    print(results["test_metrics_df"].to_string(index=False))

    print(f"\nWinner: {results['best_model_name']}")
    print(f"Optimal Parameters: {results['best_params']}")

    # Build unique product catalog for recommendation queries
    product_catalog = df_clean.groupby("product_id").agg({
        "product_category": "first",
        "product_price": "median"
    }).reset_index()

    # Build customer browsing profiles
    customer_profiles = df_clean.groupby("customer_id").agg({
        "previous_purchases": "max",
        "views": "mean",
        "cart_adds": "mean",
        "product_category": lambda s: s.mode()[0] if not s.empty else "Electronics"
    }).reset_index()

    # Customer past purchase mapping for Upgrade 4 (Exclude already purchased)
    purchases_df = df_clean[df_clean["purchases"] > 0]
    customer_purchases = {}
    for cid, group in purchases_df.groupby("customer_id"):
        customer_purchases[cid] = list(group["product_id"].unique())

    artifacts_to_save = {
        "best_model_name": results["best_model_name"],
        "best_pipeline": results["best_pipeline"],
        "best_params": results["best_params"],
        "cv_results_df": results["cv_results_df"],
        "metrics_df": results["test_metrics_df"],
        "roc_curves": results["roc_curves"],
        "confusion_matrix_data": results["confusion_matrix_data"],
        "feature_importance_df": results["feature_importance_df"],
        "cleaning_stats": cleaning_stats,
        "product_catalog": product_catalog,
        "customer_profiles": customer_profiles,
        "customer_purchases": customer_purchases,
        "cleaned_df": df_clean
    }

    joblib.dump(artifacts_to_save, "model_artifacts.pkl")
    print("\nArtifacts successfully updated and saved to 'model_artifacts.pkl'!")


if __name__ == "__main__":
    run_pipeline()
