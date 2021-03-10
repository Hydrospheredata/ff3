import logging
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from ff3.drift.test import (
    StatisticalTest, mean_test_message, median_test_message, 
    variance_test_message, chi_square_message, unique_values_test_message
)
from ff3.drift.utils import is_evaluated, not_restored


class ProfileTypes(Enum):
    CATEGORICAL = "CATEGORICAL"
    NUMERICAL = "NUMERICAL"
    UNKNOWN = "UNKNOWN"


class FeatureReportFactory:
    @classmethod
    def create(cls, name: str, s1: pd.Series, s2: pd.Series, p_threshold: float) -> 'FeatureReport':
        if cls.__is_categorical(s1):
            logging.info(f"Feature {name} is inferred as Categorical.")
            return CategoricalFeatureReport(name, s1.values, s2.values, p_threshold)
        elif cls.__is_numerical(s1):
            logging.info(f"Feature {name} is inferred as Numerical.")
            return NumericalFeatureReport(name, s1.values, s2.values, p_threshold)
        else:
            logging.info(f"Couldn't infer profile type of feature {name}.")
            return UnknownFeatureReport(name, s1.values, s2.values, p_threshold)

    @staticmethod
    def __is_categorical(sr: pd.Series) -> bool:
        is_categorical = pd.api.types.is_categorical_dtype(sr)
        unique_values = sr.value_counts().shape[0]
        is_categorical_fallback = unique_values / sr.shape[0] <= 0.05 and unique_values <= 20
        return is_categorical or is_categorical_fallback

    @staticmethod
    def __is_numerical(sr: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(sr)

    @staticmethod
    def from_dict(name: str, in_dict: dict) -> 'FeatureReport':
        if in_dict["profile"] == ProfileTypes.NUMERICAL:
            return NumericalFeatureReport.from_dict(name, in_dict)
        elif in_dict["profile"] == ProfileTypes.CATEGORICAL:
            return CategoricalFeatureReport.from_dict(name, in_dict)
        else: 
            return UnknownFeatureReport.from_dict(name, in_dict)
        
        
class FeatureReport(ABC):
    def __init__(self, name: str, a1: np.array, a2: np.array, p_threshold: float):
        self._is_evaluated = False
        self._is_restored = False

        self.name = name
        self.a1 = a1
        self.a2 = a2
        self.tests = []
        self.drift_probability = 0
        self.p_threshold = p_threshold

        self.bins = []
        self.a1_hist_values = []
        self.a2_hist_values = []

    def __repr__(self):
        return f"Feature report for {self.name}"

    def _prepare(self):
        self.bins, self.a1_hist_values, self.a2_hist_values = self._get_histogram()
    
    def _combine(self, other: 'FeatureReport'):
        """
        Combine two statistical feature reports from submitted features respectively 
        to calculate the conditional distribution X2|X1.
        """
        return BivariateReportFactory.create(self, other)

    @abstractmethod
    def _get_histogram(self) -> Tuple[np.array, np.array, np.array]:
        """
        Create a histogram for the given feature report.
        """
        pass

    @not_restored
    def eval(self):
        self._prepare()

    @is_evaluated
    def to_dict(self) -> dict:
        return {
            "hist": {
                "bins": self.bins,
                "a1": self.a1_hist_values,
                "a2": self.a2_hist_values
            },
            "tests": {
                test.name : test.to_dict() 
                for test in self.tests
            },
            "drift_probability": self.drift_probability,
            "p_threshold": self.p_threshold,
        }
    
    @property
    @is_evaluated
    def has_failed_tests(self) -> bool:
        return any(t.has_changed for t in self.tests)
    
    @property
    @is_evaluated
    def is_drifted(self) -> bool:
        return self.drift_probability > 0.75

    @is_evaluated
    def get_warning(self) -> Optional[dict]:
        if self.is_drifted:
            return {"drift_probability_per_feature": self.drift_probability,
                    "message": f"The feature {self.name} has drifted. Following statistics have changed: {[x.name for x in self.tests if x.has_changed]}."}

    @classmethod
    def from_dict(cls, name: str, in_dict: dict) -> 'FeatureReport':
        instance = cls(name, None, None, in_dict["p_threshold"])
        instance._is_evaluated = True
        instance._is_restored = True
        instance.bins = in_dict["hist"]["bins"]
        instance.a1_hist_values = in_dict["hist"]["a1"]
        instance.a2_hist_values = in_dict["hist"]["a2"]
        instance.tests = [
            StatisticalTest.from_dict(name, test)
            for name, test in in_dict["tests"].items()
        ]
        instance.drift_probability = in_dict["drift_probability"]
        return instance


class UnknownFeatureReport(FeatureReport):
    def __init__(self, name: str, a1: np.array, a2: np.array, p_threshold: float):
        super().__init__(name, a1, a2, p_threshold)
        self._is_evaluated = True

    def to_dict(self):
        result = super().to_dict()
        result.update({"profile": ProfileTypes.UNKNOWN.value})
        return result
    
    def _get_histogram(self):
        return [], [], []


class NumericalFeatureReport(FeatureReport):
    def __init__(self, name: str, a1: np.array, a2: np.array, p_threshold: float):
        super().__init__(name, a1, a2, p_threshold)

        # List of tests used for comparing a1 and a2 numerical columns
        self.tests: List[StatisticalTest] = [
            StatisticalTest("Mean", np.mean, stats.ttest_ind, mean_test_message, self.p_threshold, {"equal_var": False}),
            StatisticalTest("Median", np.median, stats.median_test, median_test_message, self.p_threshold, {"ties": "ignore"}),
            StatisticalTest("Variance", np.var, stats.levene, variance_test_message, self.p_threshold, {"center": "mean"}),
        ]

    @not_restored
    def eval(self):
        super().eval()
        for test in self.tests:
            test.eval(self.a1, self.a2)

        # TODO add KS test to numerical features?
        # _, p_value = stats.ks_2samp(self.a1, self.a2)
        # self.ks_test_change = p_value <= 0.05

        self._is_evaluated = True
        self.drift_probability = np.mean([test.has_changed for test in self.tests])

    def _get_histogram(self):
        a1 = self.a1.astype(float)
        a2 = self.a2.astype(float)

        data_minimum = min(a1.min(), a2.min())
        data_maximum = max(a1.max(), a2.max())

        a1_hist_values, bin_edges = np.histogram(a1,
                                          bins='fd',
                                          range=[data_minimum, data_maximum])

        # Cap maximum number of histograms due to UI and human perception limitations
        if len(bin_edges) > 40:
            a1_hist_values, bin_edges = np.histogram(a1,
                                              bins=40,
                                              range=[data_minimum, data_maximum])

        a2_hist_values, _ = np.histogram(a2,
                                  bins=bin_edges,
                                  range=[data_minimum, data_maximum])

        # Obtain PMF for binned features. np.hist returns PDF which could be less 
        # recognizable by non-data scientists
        a1_hist_values = a1_hist_values / a1_hist_values.sum()
        a2_hist_values = a2_hist_values / a2_hist_values.sum()

        return bin_edges, a1_hist_values, a2_hist_values

    def to_dict(self):
        result = super().to_dict()
        result.update({"profile": ProfileTypes.NUMERICAL.value})
        return result
        

class CategoricalFeatureReport(FeatureReport):
    def __init__(self, name: str, a1: np.array, a2: np.array, p_threshold: float):
        super().__init__(name, a1, a2, p_threshold)

        # List of tests used for comparing a1 and a2 categorical frequencies
        self.tests: List[StatisticalTest] = [
            StatisticalTest("Category densities", lambda x: np.round(x, 3), self.__chisquare, chi_square_message, self.p_threshold),
            StatisticalTest("Unique Values", lambda density: self.bins[np.nonzero(density)], self.__unique_values_test, unique_values_test_message, self.p_threshold),
        ]

    def __unique_values_test(self, a1_density, a2_density):
        # If we have categories with positive frequencies in a2, but have no such categories in a1
        if sum((a1_density > 0) & (a2_density == 0)) > 0:
            return None, 0  # Definitely Changed
        else:
            return None, 1  # Prob. not changed

    def __chisquare(self, a1_density, a2_density):
        a2_sample_size = self.a2.shape[0]
        # ChiSquare test compares Observed Frequencies to Expected Frequencies, so we need to change arguments placement
        return stats.chisquare(np.round(a2_density * a2_sample_size) + 1,
                               np.round(a1_density * a2_sample_size) + 1)

    def _get_histogram(self):
        a1_categories, t_counts = np.unique(self.a1, return_counts=True)
        a2_categories, p_counts = np.unique(self.a2, return_counts=True)

        # Calculate superset of categories
        common_categories = np.array(list(set(a1_categories).union(set(a2_categories))))
        common_categories.sort()

        a1_category_to_count = dict(zip(a1_categories, t_counts))
        a2_category_to_count = dict(zip(a2_categories, p_counts))

        # Calculate frequencies per category for a1 and a2
        a1_counts_for_common_categories = np.array(
            [a1_category_to_count.get(category, 0) for category in common_categories])
        a2_counts_for_common_categories = np.array(
            [a2_category_to_count.get(category, 0) for category in common_categories])

        # Normalize frequencies to density
        a1_density = a1_counts_for_common_categories / a1_counts_for_common_categories.sum()
        a2_density = a2_counts_for_common_categories / a2_counts_for_common_categories.sum()

        return common_categories, a1_density, a2_density

    def eval(self):
        super().eval()
        for test in self.tests:
            test.eval(self.a1_hist_values, self.a2_hist_values)

        self._is_evaluated = True
        self.drift_probability = np.mean([test.has_changed for test in self.tests])

    def to_dict(self):
        result = super().to_dict()
        result.update({"profile": ProfileTypes.CATEGORICAL.value})
        return result


class BivariateReportFactory:
    @classmethod
    def create(cls, fr1: FeatureReport, fr2: FeatureReport):
        return BivariateFeatureReport(
            fr1.name, *cls.__encode_feature_report(fr1),
            fr2.name, *cls.__encode_feature_report(fr2)
        )

    @classmethod
    def __encode_feature_report(cls, fr: FeatureReport):
        if isinstance(fr, NumericalFeatureReport):
            labels, a1, a2 = cls.__discretize_numerical_report(fr)
        elif isinstance(fr, CategoricalFeatureReport):
            labels, a1, a2 = cls.__encode_categorical_report(fr)
        else:
            labels = np.zeros(len(fr.bins))
            a1 = a2 = np.zeros(len(fr.a1_hist_values))
        return labels, a1, a2

    @staticmethod
    def __encode_categorical_report(feature_report: 'CategoricalFeatureReport'):
        labels = feature_report.bins

        encoder = LabelEncoder()
        encoder.classes_ = np.array(labels)

        a1 = encoder.transform(feature_report.a1).flatten()
        a2 = encoder.transform(feature_report.a2).flatten()
        return labels, a1, a2

    @staticmethod
    def __discretize_numerical_report(feature_report: NumericalFeatureReport):
        bin_edges = feature_report.bins

        discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
        discretizer.bin_edges_ = [np.array(bin_edges)]
        discretizer.n_bins_ = np.array([len(bin_edges) - 1])

        labels = np.array([f"{b1:.2f} <= {b2:.2f}" for b1, b2 in zip(bin_edges, bin_edges[1:])])

        return labels, discretizer.transform(feature_report.a1.reshape(-1, 1)).flatten(), \
               discretizer.transform(feature_report.a2.reshape(-1, 1)).flatten()


class BivariateFeatureReport:
    def __init__(self, f1_name: str, f1_labels: List[str], f1_a1: np.array, f1_a2: np.array,
                 f2_name: str, f2_labels: List[str], f2_a1: np.array, f2_a2: np.array):
        """
        Parameters
        ----------
        f1_name: Name of a first feature
        f1_labels: List of human-readable labels for constructing a heatmap
        f1_a1: Ordinally encoded array of a1 labels from the first feature
        f1_a2: Ordinally encoded array of a2 labels from the first feature
        f2_name: Name of a second feature
        f2_labels: List of human-readable labels for constructing a heatmap
        f2_a1: Ordinally encoded array of a1 labels from the second feature
        f2_a2: Ordinally encoded array of a2 labels from the second feature
        """
        self.f1_name = f1_name
        self.f1_labels = f1_labels
        self.f1_a1 = f1_a1
        self.f1_a2 = f1_a2

        self.f2_name = f2_name
        self.f2_labels = f2_labels
        self.f2_a1 = f2_a1
        self.f2_a2 = f2_a2

        self.a1_heatmap: HeatMapData = None
        self.a2_heatmap: HeatMapData = None

        # Todo specify is ordinal or is categorical?!
        # if ordinal-ordinal, then KS-test is used
        # if categorical-categorical, then chisquare test is used
        self.drifted: bool = False
        self._is_evaluated = False

    def __repr__(self):
        return f"Feature report for {self.f1_name}|{self.f2_name}"

    def eval(self):
        # TODO calculate GOF here?
        self.a1_heatmap = HeatMapData(x_title=self.f1_name, y_title=self.f2_name,
                                      x_labels=self.f1_labels, y_labels=self.f2_labels,
                                      x=self.f1_a1, y=self.f2_a1)
        self.a2_heatmap = HeatMapData(x_title=self.f1_name, y_title=self.f2_name,
                                      x_labels=self.f1_labels, y_labels=self.f2_labels,
                                      x=self.f1_a2, y=self.f2_a2)
        self._is_evaluated = True


class HeatMapData:
    def __init__(self, x_title: str, y_title: str,
                 x_labels: np.array, y_labels: np.array,
                 x: np.array, y: np.array):
        """
        Container for heatmap data to plot on the UI. Calculates densities between 
        ordinally encoded labels in x and y correspondingly
        Parameters
        ----------
        x_title x axis name
        y_title y axis name
        x_labels  list of human readable x labels
        y_labels  list of human readable y labels
        x Ordinaly encoded x
        y Ordinaly encoded y
        """
        self.x_title = x_title
        self.y_title = y_title
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.x = x
        self.y = y

        # Computes heatmap density
        intensity_list = []
        for ordinal_label_y, _ in enumerate(y_labels):
            for ordinal_label_x, _ in enumerate(x_labels):
                x_mask = x == ordinal_label_x
                if x_mask.sum() > 0:
                    y_mask = y[x_mask] == ordinal_label_y
                    intensity = np.round(y_mask.mean(), 3)
                else:
                    intensity = 0
                intensity_list.append(intensity)

        self.intensity = np.array(intensity_list).reshape(len(y_labels), len(x_labels))
