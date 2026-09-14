"""Nodos del pipeline 'retrasos_ml'.

Modelo tactico de riesgo de retraso para el aeropuerto SCTE (Puerto Montt - El Tepual).
Horizonte de prediccion: 3 horas (t+3).

Feature: condiciones meteorologicas en el momento actual (hora t).
Target: condicion adversa en t+3 (viento, rafagas, precipitacion o WMO adverso).

Split temporal: train 2020-2024 / score 2025-2026.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

log = logging.getLogger(__name__)

_RANDOM_STATE = 42
_HORIZONTE_HORAS = 3

_UMBRAL_VIENTO_KMH = 25.0
_UMBRAL_RAFAGA_KMH = 35.0
_UMBRAL_PRECIP_MM = 5.0
_CODIGOS_ADVERSOS = {55, 61, 63, 65, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}
_ANIO_CORTE_SCORE = 2025


def _condicion_adversa(df: pd.DataFrame) -> pd.Series:
    return (
        (df["veloc_viento"] > _UMBRAL_VIENTO_KMH)
        | (df["rafagas_viento"] > _UMBRAL_RAFAGA_KMH)
        | (df["precipitacion"] > _UMBRAL_PRECIP_MM)
        | (df["codigo_clima"].isin(_CODIGOS_ADVERSOS))
    ).astype("int8")


def preparar_datos_scte(vuelos_raw: pd.DataFrame, clima_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra vuelos SCTE y une con clima usando horizonte t+3."""
    # --- Clima horario con target t+3 ---
    clima = clima_raw.copy()
    clima["dt"] = pd.to_datetime(clima["fecha_hora"]).dt.tz_localize(None)
    clima = clima.sort_values("dt").reset_index(drop=True)

    # Features en t: todas las variables climaticas actuales
    clima["hora_dt"] = clima["dt"].dt.floor("h")

    # Target en t+3: condicion adversa futura
    clima["adverso_t3"] = _condicion_adversa(clima).shift(-_HORIZONTE_HORAS)

    # --- Vuelos SCTE ---
    vuelos = vuelos_raw[vuelos_raw["aeropuerto_oaci"] == "SCTE"].copy()
    log.info("Vuelos SCTE totales: %d", len(vuelos))

    vuelos["hora_dt"] = (
        pd.to_datetime(vuelos["dt_operacion"], utc=True)
        .dt.tz_convert("America/Santiago")
        .dt.tz_localize(None)
        .dt.floor("h")
    )

    # Restringir al periodo con datos de clima
    anio_min = clima["hora_dt"].dt.year.min()
    vuelos = vuelos[vuelos["hora_dt"].dt.year >= anio_min].copy()
    log.info("Vuelos SCTE periodo clima (%d+): %d", anio_min, len(vuelos))

    # JOIN: features de clima en t + target en t+3
    clima_join = clima.drop(columns=["fecha_hora", "dt"])
    df = vuelos.merge(clima_join, on="hora_dt", how="left")
    log.info("Cobertura clima: %.2f%%", df["temperatura"].notna().mean() * 100)

    # Feature engineering temporal
    df["hora"] = df["hora_dt"].dt.hour
    df["mes"] = df["hora_dt"].dt.month
    df["anio"] = df["hora_dt"].dt.year
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["es_aterrizaje"] = (df["tipo_operacion"] == "A").astype("int8")

    # Split temporal
    df["split"] = df["anio"].apply(
        lambda y: "score" if y >= _ANIO_CORTE_SCORE else "train"
    )

    log.info(
        "Split: train=%d | score=%d",
        (df["split"] == "train").sum(),
        (df["split"] == "score").sum(),
    )
    log.info(
        "Tasa adversa t+3 — train: %.1f%% | score: %.1f%%",
        df.loc[df["split"] == "train", "adverso_t3"].mean() * 100,
        df.loc[df["split"] == "score", "adverso_t3"].mean() * 100,
    )
    return df


def entrenar_modelo_retrasos(
    scte_vuelos_clima: pd.DataFrame,
) -> tuple[pd.DataFrame, RandomForestClassifier]:
    """Entrena RF para predecir condicion adversa en t+3 usando datos 2020-2024."""
    features = [
        "temperatura",
        "precipitacion",
        "veloc_viento",
        "rafagas_viento",
        "codigo_clima",
        "humedad",
        "hora_sin",
        "hora_cos",
        "mes_sin",
        "mes_cos",
        "es_aterrizaje",
    ]

    train = scte_vuelos_clima[scte_vuelos_clima["split"] == "train"].dropna(
        subset=features + ["adverso_t3"]
    )
    test = scte_vuelos_clima[scte_vuelos_clima["split"] == "score"].dropna(
        subset=features + ["adverso_t3"]
    )

    X_train, y_train = train[features], train["adverso_t3"]
    X_test, y_test = test[features], test["adverso_t3"]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RANDOM_STATE)
    cv_auc = cross_val_score(rf, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

    metricas = pd.DataFrame(
        [
            {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc_test": round(roc_auc_score(y_test, y_prob), 4),
                "roc_auc_cv5_mean": round(cv_auc.mean(), 4),
                "roc_auc_cv5_std": round(cv_auc.std(), 4),
                "train_anios": "2020-2024",
                "score_anios": f"{_ANIO_CORTE_SCORE}-2026",
                "horizonte_horas": _HORIZONTE_HORAS,
            }
        ]
    )
    log.info(
        "Metricas retrasos_ml (t+%dh, test 2025-2026):\n%s",
        _HORIZONTE_HORAS,
        metricas.to_string(index=False),
    )
    return metricas, rf


def score_operativo(
    scte_vuelos_clima: pd.DataFrame,
    modelo_retrasos_scte: RandomForestClassifier,
) -> pd.DataFrame:
    """Genera tabla de score de riesgo con Alerta_Operativa (periodo 2025-2026)."""
    features = [
        "temperatura",
        "precipitacion",
        "veloc_viento",
        "rafagas_viento",
        "codigo_clima",
        "humedad",
        "hora_sin",
        "hora_cos",
        "mes_sin",
        "mes_cos",
        "es_aterrizaje",
    ]

    df = scte_vuelos_clima[scte_vuelos_clima["split"] == "score"].copy()
    mask = df[features].notna().all(axis=1)
    X = df.loc[mask, features]

    prob = modelo_retrasos_scte.predict_proba(X)[:, 1]
    df.loc[mask, "prob_retraso_t3"] = prob

    def alerta(p: float) -> str:
        if p < 0.30:
            return "Verde (Operacion Normal)"
        if p <= 0.70:
            return "Amarillo (Monitoreo Preventivo)"
        return "Rojo (Alerta de Retraso)"

    df.loc[mask, "Alerta_Operativa"] = df.loc[mask, "prob_retraso_t3"].apply(alerta)

    cols_salida = [
        "hora_dt",
        "anio",
        "mes",
        "hora",
        "tipo_operacion",
        "numero_vuelo",
        "aerolinea_dgac",
        "modelo_avion",
        "es_internacional",
        "temperatura",
        "precipitacion",
        "veloc_viento",
        "rafagas_viento",
        "humedad",
        "codigo_clima",
        "adverso_t3",
        "prob_retraso_t3",
        "Alerta_Operativa",
    ]
    resultado = df[cols_salida].sort_values("hora_dt").reset_index(drop=True)

    dist = resultado["Alerta_Operativa"].value_counts()
    log.info("Distribucion Alerta_Operativa (2025-2026):\n%s", dist.to_string())
    return resultado
