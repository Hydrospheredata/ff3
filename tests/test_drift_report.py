from typing import Tuple
import datetime

import pytest
import pandas as pd
import numpy as np

from ff3 import DriftReport
from ff3.drift.features import (
    NumericalFeatureReport, CategoricalFeatureReport, UnknownFeatureReport
)
from ff3.drift.errors import (
    ValidationError, EvalOnRestoredError, NotEvaluatedError
)

def filter_features(t, drift):
    return list(filter(
        lambda x: isinstance(x, t), 
        drift.feature_reports.values()
    ))


@pytest.fixture    
def nlines() -> int:
    return 150


@pytest.fixture
def data(nlines: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame({
        "categorical": np.random.choice(np.array([1, 2, 3]), nlines),
        "numerical": np.random.uniform(0, 1, nlines),
        "unknown": list(datetime.datetime.now() for _ in range(nlines))
    })
    return df, df.copy()


@pytest.fixture
def drift(data) -> DriftReport:
    drift = DriftReport(*data, p_threshold=.02)
    drift.eval()
    return drift


def test_de_serialization(drift: DriftReport):
    deserialized = DriftReport.from_dict(drift.to_dict())

    s_feature_names = [r.name for r in drift.feature_reports.values()]
    d_feature_names = [r.name for r in deserialized.feature_reports.values()]
    assert s_feature_names == d_feature_names

    s_bins = np.array([r.bins for r in drift.feature_reports.values()], dtype=object)
    d_bins = np.array([r.bins for r in deserialized.feature_reports.values()], dtype=object)
    assert s_bins.shape == d_bins.shape

    assert np.round(drift.odp, 4) == np.round(deserialized.odp, 4)
    assert drift.p_threshold == deserialized.p_threshold 


def test_feature_inference(drift: DriftReport):
    ns = filter_features(NumericalFeatureReport, drift)
    cs = filter_features(CategoricalFeatureReport, drift)
    us = filter_features(UnknownFeatureReport, drift)

    assert len(ns) == 1
    assert len(cs) == 1
    assert len(us) == 1

    n, c, u = ns[0], cs[0], us[0]
    assert n.name == "numerical"
    assert c.name == "categorical"
    assert u.name == "unknown"


def test_bivariate_report_generation(drift: DriftReport):
    n = filter_features(NumericalFeatureReport, drift)[0]
    c = filter_features(CategoricalFeatureReport, drift)[0]
    br = c._combine(n)

    fig = drift.bivariate_report("categorical", "numerical")
    assert fig.layout.title.text == repr(br)
    assert fig.layout.xaxis.title.text == "categorical"
    assert fig.layout.yaxis.title.text == "numerical"


def test_detailed_report_generation(drift: DriftReport):
    fig = drift.detailed_report()
    plotted_labels = set([button.label for button in fig.layout.updatemenus[0].buttons])
    report_labels = set(drift.feature_reports.keys())
    assert len(plotted_labels.symmetric_difference(report_labels)) == 0


def test_data_validation():
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"c": [3], "d": [4]})
    with pytest.raises(ValidationError):
        drift = DriftReport(df1, df2)
        drift.eval()


def test_eval_on_restored_raises_an_error(drift: DriftReport):
    serialized = drift.to_dict()
    with pytest.raises(EvalOnRestoredError):
        restored = DriftReport.from_dict(serialized)
        restored.eval()


def test_not_evaled_raises_an_error(data: Tuple[pd.DataFrame, pd.DataFrame]):
    d1, d2 = data
    drift = DriftReport(d1, d2)
    
    with pytest.raises(NotEvaluatedError):
        serialized = drift.to_dict()

    with pytest.raises(NotEvaluatedError):
        artifact = drift.overall_report()
    
    with pytest.raises(NotEvaluatedError):
        artifact = drift.detailed_report()
    
    with pytest.raises(NotEvaluatedError):
        artifact = drift.bivariate_report("numerical", "categorical")
