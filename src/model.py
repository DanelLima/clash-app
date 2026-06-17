from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES: list[str] = [
    "player_starting_trophies",
    "opponent_starting_trophies",
    "trophy_difference",
    "battle_hour",
    "crown_difference",
]

TARGET: str = "win_binary"

# Hiperparâmetros padrão do RandomForest
RF_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators":      300,
    "max_depth":         12,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "class_weight":      "balanced",   # lida com desbalanceamento de classes
    "random_state":      42,
    "n_jobs":            -1,           # usa todos os núcleos disponíveis
}


@dataclass
class ModelMetrics:
    """Contém todas as métricas de avaliação do modelo treinado.

    Attributes:
        accuracy:           Acurácia global (proporção de predições corretas).
        precision:          Precisão para a classe positiva (win).
        recall:             Recall para a classe positiva (win).
        f1:                 F1-score para a classe positiva (win).
        cv_mean:            Média do F1 na validação cruzada (5-fold).
        cv_std:             Desvio padrão do F1 na validação cruzada.
        confusion_matrix:   Matriz de confusão [[TN, FP], [FN, TP]].
        classification_report: Relatório completo como string.
        feature_importances: Importância de cada feature (dict carta → valor).
        test_size:          Tamanho do conjunto de teste (número de amostras).
        train_size:         Tamanho do conjunto de treino (número de amostras).
    """
    accuracy:               float
    precision:              float
    recall:                 float
    f1:                     float
    cv_mean:                float
    cv_std:                 float
    confusion_matrix:       np.ndarray
    classification_report:  str
    feature_importances:    dict[str, float]
    test_size:              int
    train_size:             int

    def summary(self) -> str:
        """Retorna um resumo formatado das principais métricas.

        Returns:
            String multilinha com as métricas principais.
        """
        lines = [
            "=" * 52,
            "           MÉTRICAS DO MODELO",
            "=" * 52,
            f"  Acurácia         : {self.accuracy:.4f}  ({self.accuracy * 100:.2f}%)",
            f"  Precisão         : {self.precision:.4f}  ({self.precision * 100:.2f}%)",
            f"  Recall           : {self.recall:.4f}  ({self.recall * 100:.2f}%)",
            f"  F1-score         : {self.f1:.4f}  ({self.f1 * 100:.2f}%)",
            f"  CV F1 (5-fold)   : {self.cv_mean:.4f} ± {self.cv_std:.4f}",
            "-" * 52,
            f"  Treino           : {self.train_size:,} amostras",
            f"  Teste            : {self.test_size:,} amostras",
            "=" * 52,
            "\n  Importância das Features:",
        ]
        for feat, imp in sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(imp * 40)
            lines.append(f"  {feat:<35} {imp:.4f}  {bar}")
        lines.append("=" * 52)
        return "\n".join(lines)


@dataclass
class TrainingResult:
    """Resultado completo do treinamento, incluindo modelo e métricas.

    Attributes:
        model:    Pipeline sklearn com scaler + RandomForestClassifier treinado.
        metrics:  Objeto ModelMetrics com todas as métricas de avaliação.
        features: Lista de features utilizadas no treinamento.
        target:   Nome da coluna alvo.
    """
    model:    Pipeline
    metrics:  ModelMetrics
    features: list[str] = field(default_factory=lambda: FEATURES.copy())
    target:   str = TARGET


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Valida que o DataFrame possui todas as colunas necessárias e dados suficientes.

    Args:
        df: DataFrame a ser validado.

    Raises:
        KeyError:   Se alguma feature ou o target estiverem ausentes.
        ValueError: Se o DataFrame tiver menos de 100 amostras ou apenas uma classe.
    """
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colunas ausentes no DataFrame: {missing}. "
            "Execute o pipeline preprocessing → feature_engineering antes de train_model()."
        )

    if len(df) < 100:
        raise ValueError(
            f"DataFrame com apenas {len(df)} amostras. "
            "São necessárias pelo menos 100 amostras para treinar o modelo."
        )

    unique_classes = df[TARGET].nunique()
    if unique_classes < 2:
        raise ValueError(
            f"A coluna '{TARGET}' contém apenas {unique_classes} classe(s). "
            "São necessárias pelo menos 2 classes para classificação binária."
        )


def _build_pipeline(rf_params: dict[str, Any]) -> Pipeline:
    """Constrói o pipeline sklearn com StandardScaler e RandomForestClassifier.

    Embora o RandomForest não seja sensível à escala, o scaler garante
    compatibilidade com possíveis substituições de estimador no futuro.

    Args:
        rf_params: Hiperparâmetros para o RandomForestClassifier.

    Returns:
        Pipeline sklearn configurado e pronto para treino.
    """
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", RandomForestClassifier(**rf_params)),
    ])


def _compute_metrics(
    pipeline:   Pipeline,
    X_train:    pd.DataFrame,
    X_test:     pd.DataFrame,
    y_train:    pd.Series,
    y_test:     pd.Series,
) -> ModelMetrics:
    """Calcula todas as métricas de avaliação sobre o conjunto de teste.

    Args:
        pipeline: Pipeline treinado.
        X_train:  Features de treino (para validação cruzada).
        X_test:   Features de teste.
        y_train:  Target de treino (para validação cruzada).
        y_test:   Target de teste (ground truth).

    Returns:
        ModelMetrics com todas as métricas calculadas.
    """
    y_pred = pipeline.predict(X_test)

    # Validação cruzada no conjunto de treino (evita data leakage)
    logger.info("Executando validação cruzada (5-fold) no conjunto de treino...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring="f1", n_jobs=-1,
    )

    importances: dict[str, float] = dict(
        zip(
            FEATURES,
            pipeline.named_steps["classifier"].feature_importances_,
        )
    )

    return ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred,
            target_names=["loss/draw (0)", "win (1)"],
            zero_division=0,
        ),
        feature_importances=importances,
        test_size=len(y_test),
        train_size=len(y_train),
    )

def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    rf_params: dict[str, Any] | None = None,
    stratify: bool = True,
) -> TrainingResult:
    """Treina um RandomForestClassifier para prever vitórias em partidas de Clash Royale.

    Etapas internas:
        1. Validação do DataFrame de entrada.
        2. Separação de features (X) e target (y).
        3. Divisão treino/teste com train_test_split estratificado.
        4. Construção do pipeline (StandardScaler + RandomForestClassifier).
        5. Treinamento do pipeline.
        6. Cálculo de métricas no conjunto de teste.
        7. Validação cruzada (5-fold F1) no conjunto de treino.

    Args:
        df:         DataFrame enriquecido (saída de ``feature_engineering.create_features``).
        test_size:  Proporção do dataset reservada para teste (padrão: 0.2 = 20%).
        rf_params:  Hiperparâmetros customizados para o RandomForestClassifier.
                    Se None, usa RF_DEFAULT_PARAMS.
        stratify:   Se True, mantém a proporção de classes no split (padrão: True).

    Returns:
        TrainingResult com:
            - ``model``   : Pipeline sklearn treinado (predict-ready).
            - ``metrics`` : ModelMetrics com accuracy, precision, recall, f1,
                            cv_mean, cv_std, confusion_matrix, classification_report
                            e feature_importances.
            - ``features``: Lista de features utilizadas.
            - ``target``  : Nome da coluna alvo.

    Raises:
        KeyError:   Se alguma coluna necessária estiver ausente.
        ValueError: Se o DataFrame for pequeno demais ou com uma só classe.

    Example:
        >>> from preprocessing import preprocess
        >>> from feature_engineering import create_features
        >>> from model import train_model
        >>>
        >>> df = create_features(preprocess("data/battles.csv"))
        >>> result = train_model(df)
        >>> print(result.metrics.summary())
        >>> result.model.predict(X_new)
    """
    logger.info("=== Iniciando treinamento do modelo ===")

    # 1. Validação
    _validate_dataframe(df)
    params = rf_params if rf_params is not None else RF_DEFAULT_PARAMS.copy()

    # 2. Preparação das features — remove nulos residuais
    df_model = df[FEATURES + [TARGET]].dropna().copy()
    dropped = len(df) - len(df_model)
    if dropped:
        logger.warning("%d linha(s) removida(s) por conter NaN nas colunas do modelo.", dropped)

    X: pd.DataFrame = df_model[FEATURES]
    y: pd.Series    = df_model[TARGET].astype(int)

    logger.info(
        "Dataset: %d amostras | Features: %s | Target: '%s'",
        len(df_model), FEATURES, TARGET,
    )
    logger.info(
        "Distribuição do target — win: %d (%.1f%%) | loss/draw: %d (%.1f%%)",
        y.sum(), y.mean() * 100,
        (y == 0).sum(), (1 - y.mean()) * 100,
    )

    # 3. Split treino/teste
    stratify_col = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_col,
    )
    logger.info(
        "Split: %d treino / %d teste (%.0f%% / %.0f%%)",
        len(X_train), len(X_test),
        (1 - test_size) * 100, test_size * 100,
    )

    # 4. Construção do pipeline
    pipeline = _build_pipeline(params)
    logger.info("Pipeline configurado: StandardScaler → RandomForestClassifier(%s).", params)

    # 5. Treinamento
    logger.info("Treinando modelo...")
    pipeline.fit(X_train, y_train)
    logger.info("Treinamento concluído.")

    # 6 & 7. Métricas + validação cruzada
    metrics = _compute_metrics(pipeline, X_train, X_test, y_train, y_test)

    logger.info("=== Treinamento finalizado ===")
    logger.info(metrics.summary())

    return TrainingResult(
        model=pipeline,
        metrics=metrics,
        features=FEATURES.copy(),
        target=TARGET,
    )


def predict(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """Realiza predições em novos dados usando o modelo treinado.

    Args:
        model: Pipeline treinado retornado por ``train_model()``.
        data:  DataFrame com as mesmas features usadas no treino.
               Deve conter as colunas em ``FEATURES``.

    Returns:
        Array numpy de predições binárias (0 = derrota/empate, 1 = vitória).

    Raises:
        KeyError: Se alguma feature estiver ausente em ``data``.

    Example:
        >>> preds = predict(result.model, df_new[FEATURES])
        >>> print(preds)
        array([1, 0, 1, 1, 0])
    """
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise KeyError(f"Features ausentes para predição: {missing}.")

    return model.predict(data[FEATURES])


def predict_proba(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """Retorna as probabilidades de vitória para cada amostra.

    Args:
        model: Pipeline treinado retornado por ``train_model()``.
        data:  DataFrame com as features do modelo.

    Returns:
        Array numpy de shape (n_amostras, 2) com probabilidades
        [P(derrota), P(vitória)] para cada linha.

    Example:
        >>> proba = predict_proba(result.model, df_new[FEATURES])
        >>> win_probability = proba[:, 1]   # probabilidade de vitória
    """
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise KeyError(f"Features ausentes para predição: {missing}.")

    return model.predict_proba(data[FEATURES])

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import preprocess                 # type: ignore[import]
    from feature_engineering import create_features      # type: ignore[import]

    if len(sys.argv) < 2:
        print("Uso: python model.py <caminho_do_csv>")
        sys.exit(1)

    df_clean  = preprocess(sys.argv[1])
    df_feat   = create_features(df_clean)
    result    = train_model(df_feat)

    print("\n" + result.metrics.summary())
    print("\nRelatório de Classificação:")
    print(result.metrics.classification_report)
    print("Matriz de Confusão:")
    print(result.metrics.confusion_matrix)