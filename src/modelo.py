import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# features usadas pelo modelo
# ATENCAO: diferenca_coroas foi excluida porque e calculada APOS a batalha
# (usar ela causaria data leakage e acuracia artificial de 100%)
FEATURES = [
    "player_starting_trophies",
    "opponent_starting_trophies",
    "diferenca_trofeus",
    "hora_batalha",
    "hp_torre_ratio",
]


@dataclass
class MetricasModelo:
    """Agrupa todas as metricas de avaliacao do modelo."""
    acuracia:              float
    precisao:              float
    recall:                float
    f1:                    float
    media_cv:              float
    desvio_cv:             float
    matriz_confusao:       list
    relatorio_classificacao: str
    importancia_features:  dict
    tamanho_treino:        int
    tamanho_teste:         int


@dataclass
class ResultadoModelo:
    """Resultado completo: pipeline treinado e metricas."""
    pipeline: Pipeline
    metricas: MetricasModelo


def treinar_modelo(df: pd.DataFrame) -> ResultadoModelo:
    # garante que as features existam no dataframe
    faltando = [f for f in FEATURES if f not in df.columns]
    if faltando:
        raise ValueError(f"Features ausentes no dataframe: {faltando}")

    X = df[FEATURES].copy()
    y = df["vitoria_binaria"].copy()

    # remove linhas com nulo nas features
    mascara = X.notna().all(axis=1)
    X = X[mascara]
    y = y[mascara]

    # divisao treino / teste com seed fixo para reproducibilidade
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline: normalizacao + classificador
    pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classificador", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    pipeline.fit(X_treino, y_treino)
    logger.info("Modelo treinado com %d amostras", len(X_treino))

    # predicoes no conjunto de teste
    y_pred = pipeline.predict(X_teste)

    # validacao cruzada 5-fold no conjunto de treino
    scores_cv = cross_val_score(pipeline, X_treino, y_treino, cv=5, scoring="f1")

    # importancia de cada feature (extraida do random forest dentro do pipeline)
    floresta = pipeline.named_steps["classificador"]
    importancias = dict(zip(FEATURES, floresta.feature_importances_.round(4)))

    metricas = MetricasModelo(
        acuracia=round(accuracy_score(y_teste, y_pred), 4),
        precisao=round(precision_score(y_teste, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_teste, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_teste, y_pred, zero_division=0), 4),
        media_cv=round(scores_cv.mean(), 4),
        desvio_cv=round(scores_cv.std(), 4),
        matriz_confusao=confusion_matrix(y_teste, y_pred).tolist(),
        relatorio_classificacao=classification_report(y_teste, y_pred, target_names=["Derrota", "Vitoria"]),
        importancia_features=importancias,
        tamanho_treino=len(X_treino),
        tamanho_teste=len(X_teste),
    )

    logger.info("Acuracia: %.2f%%  |  F1: %.2f%%  |  CV media: %.2f%%",
                metricas.acuracia * 100, metricas.f1 * 100, metricas.media_cv * 100)

    return ResultadoModelo(pipeline=pipeline, metricas=metricas)
