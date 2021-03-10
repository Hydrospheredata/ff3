import logging
import itertools
import json
from typing import List, Optional
from functools import lru_cache
import pandas as pd
import numpy as np

from ff3.drift.features import FeatureReportFactory
from ff3.drift.plotting import DetailedPlotService, OverallPlotService, BivariatePlotService
from ff3.drift.utils import is_evaluated, not_restored
from ff3.drift.errors import ValidationError, NotFoundError


class DriftReport:
    def __init__(self, 
            d1: pd.DataFrame, 
            d2: pd.DataFrame, 
            p_threshold: float = .01, 
            skipcols: Optional[List[str]] = None,
            keepcols: Optional[List[str]] = None,
    ):
        """
        :param d1: pd.DataFrame to use for a statistical report.
        :param d2: pd.DataFrame to use for a statistical report.
        :param p_threshold: P-threshold to use in statistical tests.
        :param skipcols: List of column names, which should be ignored for processing.
        :param keepcols: List of column names, only which should be considered for processing.
            If empty, all columns are considered except ones, specified in skipcols. 
        """
        self._is_evaluated = False   # are defined tests evaluated?
        self._is_restored = False    # is self instance restored from a serialization format?

        self.p_threshold = p_threshold
        self.d1 = d1
        self.d2 = d2
        self.skipcols = set(skipcols) if skipcols is not None else set()
        self.keepcols = set(keepcols) if keepcols is not None else set()

    def __repr__(self):
        return "DriftReport"

    @lru_cache(maxsize=1)
    def _get_columns(self):
        candidates = common_columns(self.d1, self.d2).difference(self.skipcols)
        if len(self.keepcols) > 0:
            return candidates.intersection(self.keepcols)
        return candidates
        
    def _validate(self):
        """
        Validate submitted datasets for correctness.
        """
        logging.info("Validating datasets for correctness")
        if len(self._get_columns()) == 0:
            raise ValidationError("Couldn't find common columns between datasets to create a useful report")

    def _prepare(self):
        """
        Prepare submitted datasets for futher evaluation.
        """
        logging.info("Dropping possible NAs from the datasets")
        self.d1 = self.d1.dropna(axis=1, how="all")
        self.d2 = self.d2.dropna(axis=1, how="all")

        logging.info("Creating feature report definitions")
        self.feature_reports = {
            name : FeatureReportFactory.create(name, self.d1[name], self.d2[name], self.p_threshold) 
            for name in self._get_columns()
        }

    @not_restored
    def eval(self):
        """
        Evaluate all statistic tests, defined for feature reports.
        """
        self._validate()
        self._prepare()

        [feature_report.eval() for feature_report in self.feature_reports.values()]
        self._is_evaluated = True
    
    @not_restored
    @is_evaluated
    def bivariate_report(self, feature_name_1: str, feature_name_2: str):
        """
        Combine two statistical feature reports from submitted features respectively 
        to calculate the conditional distribution X2|X1.
        """
        try: 
            fr1 = self.feature_reports[feature_name_1]
        except KeyError as e:
            raise NotFoundError(f"Couldn't find a feature report with the name: {feature_name_1}") from e
        
        try: 
            fr2 = self.feature_reports[feature_name_2]
        except KeyError as e:
            raise NotFoundError(f"Couldn't find a feature report with the name: {feature_name_2}") from e

        bivariate = fr1._combine(fr2)
        bivariate.eval()
        return BivariatePlotService(bivariate).artifact

    @is_evaluated
    def overall_report(self):
        """
        Output overall statistical report about the two datasets.
        """
        message = OverallPlotService(self).artifact
        print(message)

    @is_evaluated
    def detailed_report(self):
        """
        Output detailed statistical report about the two datasets.
        """
        return DetailedPlotService(self).artifact
    
    @property
    @is_evaluated
    def odp(self):
        """
        Calculate an overall drift probability.
        """
        return np.mean([r.drift_probability for r in self.feature_reports.values()])
    
    @is_evaluated
    def to_dict(self):
        """
        Serialize the report into a dict.
        """
        return {
            "p_threshold": self.p_threshold,
            "feature_reports": {report.name : report.to_dict() for report in self.feature_reports.values()},
        }
    
    @is_evaluated
    def to_json(self, path: str):
        """
        Serialize the report into a Json format
        """
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, in_dict: dict):
        """
        Deserialize a report from a JSON format.
        """
        drift_report = cls(None, None, p_threshold=in_dict.get("p_threshold", 0.01))
        drift_report._is_evaluated = True
        drift_report._is_restored = True
        drift_report.feature_reports = {
            feature_name : FeatureReportFactory.from_dict(feature_name, report) 
            for feature_name, report in in_dict["feature_reports"].items()
        }
        return drift_report

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))


def common_columns(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.Index:
    return d1.columns.intersection(d2.columns)
