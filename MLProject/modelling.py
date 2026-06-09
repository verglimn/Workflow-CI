import os
import sys
import json
import argparse
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Heart Disease MLflow Training")
    parser.add_argument("--n_estimators",      type=int,   default=200)
    parser.add_argument("--max_depth",         type=int,   default=10)
    parser.add_argument("--learning_rate",     type=float, default=0.1)
    parser.add_argument("--min_samples_split", type=int,   default=2)
    parser.add_argument("--model_type",        type=str,   default="RandomForest",
                        choices=["RandomForest", "GradientBoosting"])
    parser.add_argument("--data_dir",          type=str,   default="dataset_preprocessing")
    return parser.parse_args()


# Load Data
def load_data(data_dir):
    data_dir = os.path.join(BASE_DIR, data_dir) if not os.path.isabs(data_dir) else data_dir
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    trainval = pd.concat([train, val], ignore_index=True)

    X_tv   = trainval.drop(columns=["target"])
    y_tv   = trainval["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]
    X_train = train.drop(columns=["target"])
    y_train = train["target"]

    log.info(f"TrainVal: {X_tv.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_tv, y_tv, X_test, y_test


# Build Model
def build_model(args):
    if args.model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_split=args.min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else 3,
            learning_rate=args.learning_rate,
            random_state=42,
        )


# Artifacts
def save_confusion_matrix(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(
        ax=ax, cmap="Blues"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_proba, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_feature_importance(model, feature_names, path):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# Main
def main():
    args = parse_args()

    active_run = mlflow.active_run()

    log.info(f"Model type   : {args.model_type}")
    log.info(f"n_estimators : {args.n_estimators}")
    log.info(f"max_depth    : {args.max_depth}")

    X_train, y_train, X_tv, y_tv, X_test, y_test = load_data(args.data_dir)
    model = build_model(args)

    def run_training(run):
        run_id = run.info.run_id

        mlflow.set_tags({
            "model_type" : args.model_type,
            "source"     : "ci_workflow",
            "author"     : "Rieco Edward",
            "timestamp"  : datetime.now().isoformat(),
        })

        mlflow.log_params({
            "model_type"        : args.model_type,
            "n_estimators"      : args.n_estimators,
            "max_depth"         : args.max_depth,
            "learning_rate"     : args.learning_rate,
            "min_samples_split" : args.min_samples_split,
        })

        model.fit(X_tv, y_tv)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "test_accuracy" : accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall"   : recall_score(y_test, y_pred),
            "test_f1"       : f1_score(y_test, y_pred),
        }
        if y_proba is not None:
            metrics["test_roc_auc"]  = roc_auc_score(y_test, y_proba)
            metrics["test_log_loss"] = log_loss(y_test, y_proba)

        mlflow.log_metrics(metrics)

        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, f"{args.model_type} - Test CM", cm_path)
        mlflow.log_artifact(cm_path, "plots")

        if y_proba is not None:
            roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
            save_roc_curve(y_test, y_proba, f"{args.model_type} - ROC", roc_path)
            mlflow.log_artifact(roc_path, "plots")

        fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        save_feature_importance(model, X_tv.columns.tolist(), fi_path)
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, "plots")

        report = classification_report(
            y_test, y_pred,
            target_names=["No Disease", "Disease"],
            output_dict=True,
        )
        report_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, "reports")

        signature = infer_signature(X_tv, model.predict(X_tv))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_tv.head(3),
        )

        model_pkl = os.path.join(ARTIFACT_DIR, "model.pkl")
        joblib.dump(model, model_pkl)
        mlflow.log_artifact(model_pkl, "model_pkl")

        log.info(f"test_f1      = {metrics['test_f1']:.4f}")
        log.info(f"test_roc_auc = {metrics.get('test_roc_auc', 'N/A')}")
        log.info(f"Run ID       = {run_id}")

        with open(os.path.join(BASE_DIR, "run_id.txt"), "w") as f:
            f.write(run_id)

    if active_run:
        log.info(f"Using existing active run: {active_run.info.run_id}")
        run_training(active_run)
    else:
        mlflow.set_experiment("Heart_Disease_CI")
        with mlflow.start_run() as run:
            run_training(run)

    log.info("Training complete!")


if __name__ == "__main__":
    main()
