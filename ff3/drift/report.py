import logging
import itertools
import json
from typing import List, Optional

import pandas as pd
import numpy as np

from ff3.drift.features import FeatureReportFactory
from ff3.drift.plotting import DetailedPlotService, OverallPlotService
from ff3.drift.utils import is_evaluated
from ff3.drift.errors import ValidationError


class DriftReport:
    def __init__(self, d1: pd.DataFrame, d2: pd.DataFrame, p_threshold: float = .01, skipcols: Optional[List[str]] = None):
        self._is_evaluated = False   # are defined tests are evaluated?
        self.p_threshold = p_threshold

        self.d1 = d1
        self.d2 = d2
        self.skipcols = skipcols or []

        # Combine features to create bivariate reports
        # for left, right in itertools.combinations(self.feature_reports, 2):
        #     left.combine(right)

    def validate(self):
        """
        Validate submitted datasets for correctness
        """
        logging.info("Validating datasets for correctness")
        if len(common_columns(self.d1, self.d2) - set(self.skipcols)) != 0:
            raise ValidationError("Couldn't find common columns between datasets to create a useful report")

    def prepare(self):
        logging.info("Dropping possible NAs from the datasets")
        self.d1 = self.d1.dropna(axis=1, how="all")
        self.d2 = self.d2.dropna(axis=1, how="all")

        logging.info("Creating feature report definitions")
        self.feature_reports = [
            FeatureReportFactory.create(feature, self.d1[feature], self.d2[feature], self.p_threshold) 
            for feature in common_columns(self.d1, self.d2) - set(self.skipcols)
        ]

    def eval(self):
        """
        Evaluate all statistic tests, defined for reports.
        """
        self.validate()
        self.prepare()

        [feature_report.eval() for feature_report in self.feature_reports]
        self._is_evaluated = True

    @is_evaluated
    def overall_report(self):
        """
        Output overall statistical report about the two datasets.
        """
        plot = OverallPlotService(self)
        plot.show()

    @is_evaluated
    def detailed_report(self):
        """
        Output detailed statistical report about the two datasets.
        """
        plot = DetailedPlotService(self)
        plot.show()
    
    @is_evaluated
    def overall_drift_probability(self):
        return np.mean([r.drift_probability for r in self.feature_reports])
    
    @is_evaluated
    def to_dict(self):
        """
        Serialize the report into a dict.
        """
        return {
            "p_threshold": self.p_threshold,
            "feature_reports": {report.name : report.to_dict() for report in self.feature_reports},
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
        drift_report.feature_reports = [
            FeatureReportFactory.from_dict(name, report) 
            for name, report in in_dict["feature_reports"].items()
        ]
        return drift_report

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))

    def __repr__(self):
        return "DriftReport"


def common_columns(d1: pd.DataFrame, d2: pd.DataFrame):
    return set([*d1.columns, *d2.columns])
