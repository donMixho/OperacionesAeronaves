"""Pipeline 'retrasos_ml' — Modelo tactico de riesgo de retraso para SCTE."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import entrenar_modelo_retrasos, preparar_datos_scte, score_operativo


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preparar_datos_scte,
                inputs=["vuelos_raw", "clima_raw"],
                outputs="scte_vuelos_clima",
                name="preparar_datos_scte_node",
            ),
            node(
                func=entrenar_modelo_retrasos,
                inputs="scte_vuelos_clima",
                outputs=["metricas_retrasos", "modelo_retrasos_scte"],
                name="entrenar_modelo_retrasos_node",
            ),
            node(
                func=score_operativo,
                inputs=["scte_vuelos_clima", "modelo_retrasos_scte"],
                outputs="score_riesgo_scte",
                name="score_operativo_node",
            ),
        ]
    )
