import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# Мы тестируем ML-логику "как в train.py", но без SageMaker env и без MLflow.
# Это нормальная практика для CI: быстрые юнит/смоук-тесты.


@pytest.fixture()
def tiny_dataset() -> pd.DataFrame:
    # Мини-датасет, который стабилен и не зависит от внешних файлов
    # Важно: структура как у твоего пайплайна: text + category + sample_weight (опц.)
    rows = [
        {"text": "password reset problem", "category": "account"},
        {"text": "cannot login to account", "category": "account"},
        {"text": "card payment declined", "category": "payments"},
        {"text": "refund for duplicate charge", "category": "payments"},
        {"text": "update my personal data", "category": "profile"},
        {"text": "change address in profile", "category": "profile"},
    ]
    df = pd.DataFrame(rows)
    # добавим веса, как в train.py (опционально)
    df["sample_weight"] = [2, 2, 1, 1, 1, 1]
    return df


def test_schema_expected(tiny_dataset: pd.DataFrame) -> None:
    assert "text" in tiny_dataset.columns
    assert "category" in tiny_dataset.columns
    # sample_weight может быть, а может и нет — но если есть, он должен быть числом
    if "sample_weight" in tiny_dataset.columns:
        assert np.issubdtype(tiny_dataset["sample_weight"].dtype, np.number)


def _fit_predict(df: pd.DataFrame, use_weights: bool) -> tuple[np.ndarray, np.ndarray]:
    X = df["text"].astype(str).values
    y = df["category"].astype(str).values

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=2000)),
            ("lr", LogisticRegression(C=2.0, max_iter=200, n_jobs=1)),
        ]
    )

    if use_weights and "sample_weight" in df.columns:
        clf.fit(X, y, lr__sample_weight=df["sample_weight"].values)
    else:
        clf.fit(X, y)

    y_pred = clf.predict(X)
    return y, y_pred


def test_training_predicts_and_f1_ok(tiny_dataset: pd.DataFrame) -> None:
    y_true, y_pred = _fit_predict(tiny_dataset, use_weights=False)
    f1 = f1_score(y_true, y_pred, average="weighted")
    # Минимальный sanity threshold: модель должна быть "не сломана".
    assert f1 >= 0.70


def test_training_with_sample_weights_does_not_crash(tiny_dataset: pd.DataFrame) -> None:
    y_true, y_pred = _fit_predict(tiny_dataset, use_weights=True)
    f1 = f1_score(y_true, y_pred, average="weighted")
    assert f1 >= 0.70


def test_sagemaker_safe_name_helper() -> None:
    # Юнит-тест на маленькую функцию, которая реально важна для деплоя
    from src.deployment.deploy_to_endpoint import _sm_safe_name

    assert _sm_safe_name("Hello World!") == "Hello-World"
    assert _sm_safe_name("%%%") == "mlflow-run"
    assert len(_sm_safe_name("a" * 200)) <= 63
