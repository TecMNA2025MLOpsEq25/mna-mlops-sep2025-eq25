# obesity_estimator/modeling/plot_curves.py
# -*- coding: utf-8 -*-
"""
Generación de curvas y gráficas de comparación de modelos.

Este script está pensado para ejecutarse vía DVC en el stage `model_plots`:

    PYTHONPATH=. python obesity_estimator/modeling/plot_curves.py

Hace:

1. Reconstruye el conjunto de test usando `params.yaml` y `data/processed`.
2. Carga `models/best_model.joblib` y genera (ROC ORIGINAL + PR):
   - Curvas ROC por clase.
   - Curvas PR por clase.
   - Barras macro/micro de AUC ROC y AP.
   - Matriz de confusión del mejor modelo.
3. A partir de los modelos entrenados (`models/<tipo>_best.joblib`) y de
   `reports/final_model_comparison.csv` genera:
   - Gráfico de barras de F1 macro por modelo (validación).
   - Heatmap modelo × métricas globales (validación).
   - Radar de F1 por clase:
       * uno por modelo,
       * uno con todos los modelos,
       * uno específico para hist_gb vs rf vs svc_rbf vs logreg.
4. Recalcula AUC ROC y AP en TEST para todos los modelos con `predict_proba`
   y actualiza:
   - reports/roc_auc_by_model.csv
   - reports/pr_auc_by_model.csv
5. Genera curvas ROC/PR micro-promediadas SOBREPUESAS para comparar:
   - hist_gb vs rf vs svc_rbf vs logreg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

PROJECT_ROOT = Path(".")
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_FIG_DIR = FIGURES_DIR / "models"


# -------------------------------------------------------------------------
# Utilidades para datos y configuración
# -------------------------------------------------------------------------
def _load_params(path: Path = PROJECT_ROOT / "params.yaml") -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró params.yaml en {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset(processed_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(processed_path)
    if target not in df.columns:
        raise KeyError(f"La columna target '{target}' no está en {processed_path}")
    df = df[~df[target].isna()].reset_index(drop=True)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _safe_stratify(y: pd.Series, enable: bool):
    if not enable:
        return None
    vc = y.value_counts(dropna=False)
    if len(vc) < 2 or (vc < 2).any():
        return None
    return y


def get_test_split(params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """Reconstruye el split de test con los mismos parámetros de params.yaml."""
    data_cfg = params.get("data", {})
    split_cfg = params.get("split", {})
    target_cfg = params.get("target", {})

    processed_path = data_cfg.get("processed_path", "data/processed/obesity_estimation_clean.csv")
    target_name = target_cfg.get("name", "NObeyesdad")

    X, y = _load_dataset(processed_path, target_name)
    strat = _safe_stratify(y, split_cfg.get("stratify", True))

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg.get("test_size", 0.3),
        random_state=split_cfg.get("random_state", 42),
        stratify=strat,
    )
    return X_test, y_test


# -------------------------------------------------------------------------
# Curvas ROC / PR (ORIGINALES) para el mejor modelo
# -------------------------------------------------------------------------
def compute_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calcula info para curvas ROC y PR (por clase + macro/micro)."""
    y_bin = label_binarize(y_true, classes=classes)

    roc_info: Dict[str, Any] = {"per_class": {}}
    pr_info: Dict[str, Any] = {"per_class": {}}

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        prec, rec, _ = precision_recall_curve(y_bin[:, idx], y_proba[:, idx])

        roc_info["per_class"][cls] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr),
        }
        pr_info["per_class"][cls] = {
            "precision": prec,
            "recall": rec,
            "ap": average_precision_score(y_bin[:, idx], y_proba[:, idx]),
        }

    roc_info["macro"] = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    roc_info["micro"] = roc_auc_score(y_bin, y_proba, average="micro", multi_class="ovr")

    pr_info["macro"] = average_precision_score(y_bin, y_proba, average="macro")
    pr_info["micro"] = average_precision_score(y_bin, y_proba, average="micro")

    return roc_info, pr_info


def plot_roc_pr_for_best(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Genera las curvas ROC/PR ORIGINALES y matriz de confusión
    para models/best_model.joblib (en test).
    """
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró {model_path}. Ejecuta primero el stage de entrenamiento."
        )
    model = joblib.load(model_path)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    classes = model.classes_

    roc_info, pr_info = compute_curves(y_test.to_numpy(), y_proba, classes)

    MODELS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ROC por clase (ORIGINAL)
    plt.figure(figsize=(8, 6))
    for cls in classes:
        d = roc_info["per_class"][cls]
        plt.plot(d["fpr"], d["tpr"], label=f"{cls} (AUC={d['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC por clase - mejor modelo (test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(MODELS_FIG_DIR / "roc_per_class_best.png", dpi=150)
    plt.close()

    # PR por clase (ORIGINAL)
    plt.figure(figsize=(8, 6))
    for cls in classes:
        d = pr_info["per_class"][cls]
        plt.plot(d["recall"], d["precision"], label=f"{cls} (AP={d['ap']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall por clase - mejor modelo (test)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(MODELS_FIG_DIR / "pr_per_class_best.png", dpi=150)
    plt.close()

    # Macro vs micro ROC
    plt.figure(figsize=(4, 4))
    labels = ["macro", "micro"]
    values = [roc_info["macro"], roc_info["micro"]]
    plt.bar(labels, values)
    plt.ylabel("AUC")
    plt.title("AUC ROC macro vs micro - mejor modelo (test)")
    plt.tight_layout()
    plt.savefig(MODELS_FIG_DIR / "roc_macro_micro_best.png", dpi=150)
    plt.close()

    # Macro vs micro PR
    plt.figure(figsize=(4, 4))
    labels = ["macro", "micro"]
    values = [pr_info["macro"], pr_info["micro"]]
    plt.bar(labels, values)
    plt.ylabel("Average Precision")
    plt.title("AP macro vs micro - mejor modelo (test)")
    plt.tight_layout()
    plt.savefig(MODELS_FIG_DIR / "pr_macro_micro_best.png", dpi=150)
    plt.close()

    # Matriz de confusión (puede coexistir con la que ya tenías)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False)
    plt.title("Matriz de confusión - mejor modelo (test)")
    plt.tight_layout()
    plt.savefig(MODELS_FIG_DIR / "confusion_matrix_best.png", dpi=150)
    plt.close()

    # CSVs mínimos (luego los sobreescribimos con todos los modelos)
    pd.DataFrame(
        {"model": ["best_model"], "roc_auc_macro": [roc_info["macro"]], "roc_auc_micro": [roc_info["micro"]]}
    ).to_csv(REPORTS_DIR / "roc_auc_by_model.csv", index=False)

    pd.DataFrame(
        {"model": ["best_model"], "ap_macro": [pr_info["macro"]], "ap_micro": [pr_info["micro"]]}
    ).to_csv(REPORTS_DIR / "pr_auc_by_model.csv", index=False)

    return roc_info, pr_info


# -------------------------------------------------------------------------
# Comparación global entre modelos
# -------------------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[INFO] No se encontró {path}, se omite.")
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_f1_bar_all_models(comparison_csv: Path = REPORTS_DIR / "final_model_comparison.csv") -> None:
    """Barras de F1 macro por modelo (usando métricas de validación)."""
    df = safe_read_csv(comparison_csv)
    if df.empty:
        return
    if "model" not in df.columns or "f1_macro" not in df.columns:
        print("[INFO] La tabla de comparación no tiene columnas 'model' y 'f1_macro'.")
        return

    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["f1_macro"])
    plt.xlabel("Modelo")
    plt.ylabel("F1 macro (validación)")
    plt.title("F1 macro por modelo (validación)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    MODELS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(MODELS_FIG_DIR / "f1_macro_by_model.png", dpi=150)
    plt.close()


def plot_metrics_heatmap(
    comparison_csv: Path = REPORTS_DIR / "final_model_comparison.csv",
    metrics: Optional[List[str]] = None,
) -> None:
    """Heatmap modelo × métricas globales (validación)."""
    df = safe_read_csv(comparison_csv)
    if df.empty:
        return
    if "model" not in df.columns:
        print("[INFO] La tabla de comparación no tiene columna 'model'.")
        return

    if metrics is None:
        candidates = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"]
        metrics = [m for m in candidates if m in df.columns]

    if not metrics:
        print("[INFO] No hay métricas numéricas para heatmap.")
        return

    mat = df[metrics].to_numpy()
    models = df["model"].tolist()

    fig, ax = plt.subplots(figsize=(1.5 * len(metrics) + 2, 0.5 * len(models) + 2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Métricas por modelo (validación)")
    fig.tight_layout()
    MODELS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MODELS_FIG_DIR / "metrics_heatmap.png", dpi=150)
    plt.close(fig)


def evaluate_models_on_test(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Evalúa cada modelo <tipo>_best.joblib en el mismo set de test."""
    results: Dict[str, Dict[str, Any]] = {}
    for name in model_names:
        model_path = PROJECT_ROOT / "models" / f"{name}_best.joblib"
        if not model_path.exists():
            print(f"[INFO] No se encontró {model_path}, se omite el modelo {name}.")
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        model_result: Dict[str, Any] = {"y_pred": y_pred}

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                model_result["y_proba"] = y_proba
            except Exception:
                pass

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        model_result["report"] = report

        results[name] = model_result
    return results


# -------------------------------------------------------------------------
# Radar de F1 por clase (por modelo, todos y subset)
# -------------------------------------------------------------------------
def plot_radar_f1_per_model(
    y_test: pd.Series,
    model_results: Dict[str, Dict[str, Any]],
    out_dir: Path = MODELS_FIG_DIR,
    subset: Optional[List[str]] = None,
) -> None:
    """Genera:
       - un radar por modelo,
       - un radar con todos los modelos,
       - un radar específico para un subset (ej. hist_gb, rf, svc_rbf, logreg).
    """
    import math

    labels = sorted(map(str, y_test.unique()))
    n_axes = len(labels)
    angles = [n / float(n_axes) * 2 * math.pi for n in range(n_axes)]
    angles += angles[:1]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Radar individual por modelo
    for model_name, res in model_results.items():
        report = res.get("report", {})
        values = [report.get(lbl, {}).get("f1-score", 0.0) for lbl in labels]
        values += values[:1]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"F1 por clase - {model_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"radar_f1_{model_name}.png", dpi=150)
        plt.close(fig)

    # 2) Radar conjunto con todos los modelos
    if len(model_results) >= 2:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        for model_name, res in model_results.items():
            report = res.get("report", {})
            values = [report.get(lbl, {}).get("f1-score", 0.0) for lbl in labels]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("F1 por clase - comparación de modelos")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig.savefig(out_dir / "radar_f1_all_models.png", dpi=150)
        plt.close(fig)

    # 3) Radar específico para subset (hist_gb vs rf vs svc_rbf vs logreg)
    if subset:
        existing = set(model_results.keys())
        chosen = [m for m in subset if m in existing]
        if len(chosen) >= 2:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, polar=True)
            for model_name in chosen:
                report = model_results[model_name].get("report", {})
                values = [report.get(lbl, {}).get("f1-score", 0.0) for lbl in labels]
                values += values[:1]
                ax.plot(angles, values, linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.15)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_ylim(0.0, 1.0)
            title = "F1 por clase - " + " vs ".join(chosen)
            ax.set_title(title)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            fig.tight_layout()
            fname = "radar_f1_" + "_vs_".join(chosen) + ".png"
            fig.savefig(out_dir / fname, dpi=150)
            plt.close(fig)


# -------------------------------------------------------------------------
# ROC / PR overlay (hist_gb vs rf vs svc_rbf vs logreg)
# -------------------------------------------------------------------------
def plot_roc_pr_overlay_subset(
    y_test: pd.Series,
    model_results: Dict[str, Dict[str, Any]],
    subset: List[str],
    out_dir: Path = MODELS_FIG_DIR,
) -> None:
    """
    Genera curvas ROC/PR micro-promediadas sobrepuestas para un subset
    de modelos multiclase, p.ej. hist_gb vs rf vs svc_rbf vs logreg.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = y_test.to_numpy()
    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)

    # ROC overlay
    plt.figure(figsize=(7, 6))
    for name in subset:
        res = model_results.get(name)
        if not res:
            continue
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue
        # micro-average ROC
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
        auc_micro = roc_auc_score(y_bin, y_proba, average="micro", multi_class="ovr")
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC_micro={auc_micro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC micro-promediada - hist_gb vs rf vs svc_rbf vs logreg (test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_overlay_hist_rf_svc_logreg.png", dpi=150)
    plt.close()

    # PR overlay
    plt.figure(figsize=(7, 6))
    for name in subset:
        res = model_results.get(name)
        if not res:
            continue
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue
        # micro-average PR
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
        ap_micro = average_precision_score(y_bin, y_proba, average="micro")
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP_micro={ap_micro:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR micro-promediada - hist_gb vs rf vs svc_rbf vs logreg (test)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_overlay_hist_rf_svc_logreg.png", dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# Recalcular AUC/AP en test para todos los modelos
# -------------------------------------------------------------------------
def recompute_auc_for_all_models(
    y_test: pd.Series,
    model_results: Dict[str, Dict[str, Any]],
) -> None:
    """Calcula AUC ROC y AP macro/micro en TEST para todos los modelos con proba."""
    rows_roc = []
    rows_pr = []
    y_true = y_test.to_numpy()

    classes = np.unique(y_true)
    y_bin_global = label_binarize(y_true, classes=classes)

    for name, res in model_results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue
        try:
            roc_macro = roc_auc_score(y_bin_global, y_proba, average="macro", multi_class="ovr")
            roc_micro = roc_auc_score(y_bin_global, y_proba, average="micro", multi_class="ovr")
            ap_macro = average_precision_score(y_bin_global, y_proba, average="macro")
            ap_micro = average_precision_score(y_bin_global, y_proba, average="micro")
        except Exception as e:
            print(f"[WARN] No se pudieron calcular AUC/AP para {name}: {e}")
            continue

        rows_roc.append(
            {"model": name, "roc_auc_macro": roc_macro, "roc_auc_micro": roc_micro}
        )
        rows_pr.append(
            {"model": name, "ap_macro": ap_macro, "ap_micro": ap_micro}
        )

    if rows_roc:
        pd.DataFrame(rows_roc).to_csv(REPORTS_DIR / "roc_auc_by_model.csv", index=False)
    if rows_pr:
        pd.DataFrame(rows_pr).to_csv(REPORTS_DIR / "pr_auc_by_model.csv", index=False)


# -------------------------------------------------------------------------
# Punto de entrada principal
# -------------------------------------------------------------------------
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    params = _load_params()

    # 1) Split de test único para todo
    X_test, y_test = get_test_split(params)

    # 2) Curvas ROC/PR ORIGINALES + matriz de confusión para best_model
    plot_roc_pr_for_best(X_test, y_test)

    # 3) Comparación de modelos a partir de los artefactos entrenados
    model_cfg = params.get("model", {})
    candidates = model_cfg.get("candidates", [])
    if not candidates:
        comp_df = safe_read_csv(REPORTS_DIR / "final_model_comparison.csv")
        if not comp_df.empty and "model" in comp_df.columns:
            candidates = comp_df["model"].tolist()

    model_results = evaluate_models_on_test(X_test, y_test, candidates)

    if not model_results:
        print("[INFO] No se encontraron modelos para comparación.")
        return

    # 3a) Radar F1 por clase (por modelo, todos, y subset hist_gb/rf/svc_rbf/logreg)
    subset_four = ["hist_gb", "rf", "svc_rbf", "logreg"]
    plot_radar_f1_per_model(y_test, model_results, subset=subset_four)

    # 3b) Barras de F1 macro por modelo (validación)
    plot_f1_bar_all_models()

    # 3c) Heatmap modelo × métricas globales (validación)
    plot_metrics_heatmap()

    # 3d) ROC/PR overlaid para hist_gb vs rf vs svc_rbf vs logreg
    plot_roc_pr_overlay_subset(y_test, model_results, subset_four)

    # 3e) Recalcular AUC/AP en test para todos los modelos y actualizar CSVs
    recompute_auc_for_all_models(y_test, model_results)

    print("Curvas y gráficas de comparación de modelos generadas correctamente.")


if __name__ == "__main__":
    main()
