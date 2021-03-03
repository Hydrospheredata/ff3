from typing import Callable
import numpy as np


class StatisticalTest:
    def __init__(self, name: str,
                 statistic_func: Callable,
                 statistic_test_func: Callable,
                 message_generation_func: Callable,
                 p_threshold: float,
                 statistic_test_func_kwargs=None):
        """
        Parameters
        ----------
        name - Name of the statistical test used
        statistic_func -  Function which is used to calculate tested statistic from np.array
        statistic_test_func - Function which is used to calculate (test_statistic, test_p) from 2 np.arrays
        message_generation_func - Function which us used to produce a message to be displayed on the UI based on StatisticalTest object
        statistic_test_func_kwargs - kwargs passed to statistic_test_func
        """

        self.name = name
        self.has_changed = None
        self.message_generation_func = message_generation_func
        self.message = None

        self.statistic_func = statistic_func
        self.a1_statistic = None
        self.a2_statistic = None

        self.statistic_test_func = statistic_test_func
        self.statistic_test_func_kwargs = statistic_test_func_kwargs or {}
        self.test_statistic = None
        self.test_p = None
        self.p_threshold = p_threshold

    def eval(self, a1: np.array, a2: np.array):
        self.a1_statistic = self.statistic_func(a1)
        self.a2_statistic = self.statistic_func(a2)

        try:
            test_statistic, test_p = self.statistic_test_func(a1, a2, **self.statistic_test_func_kwargs)[:2]
        except Exception as e:
            self.message = f"Unable to calculate statistic: {str(e)}"
            self.has_changed = False
        else:
            self.test_statistic = test_statistic
            self.test_p = test_p
            self.has_changed = test_p < self.p_threshold
            self.message = self.message_generation_func(self)

    def to_dict(self):
        return {
            "has_changed": bool(self.has_changed),
            "a1_statistic": self.a1_statistic,
            "a2_statistic": self.a2_statistic,
            "p_threshold": self.p_threshold,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, name: str, in_dict: dict) -> 'StatisticalTest':
        instance = cls(name, None, None, None, in_dict["p_threshold"])
        instance.a1_statistic = in_dict["a1_statistic"]
        instance.a2_statistic = in_dict["a2_statistic"]
        instance.message =      in_dict["message"]
        instance.has_changed =  in_dict["has_changed"]
        return instance


def threshold_to_apa_style(t: float):
    return str(t)[1:]


def mean_test_message(test: StatisticalTest):
    if test.has_changed:
        return f"Significant change in the mean, p<{threshold_to_apa_style(test.p_threshold)}"
    else:
        return f"No significant change in the mean"


def variance_test_message(test: StatisticalTest):
    if test.has_changed:
        return f"Significant change in the variance, p<{threshold_to_apa_style(test.p_threshold)}"
    else:
        return f"No significant change in the variance"


def median_test_message(test: StatisticalTest):
    if test.has_changed:
        return f"Significant change in the median, p<{threshold_to_apa_style(test.p_threshold)}"
    else:
        return f"No significant change in the median"


def unique_values_test_message(test: StatisticalTest):
    if test.has_changed:
        new_categories = set(test.a1_statistic).symmetric_difference(test.a2_statistic)
        return f"There are new categories {new_categories} that were observed in one dataset, but not in the other."
    else:
        return f"No change"


def chi_square_message(test: StatisticalTest):
    if test.has_changed:
        return f"A2 categorical data has different frequencies at p<{threshold_to_apa_style(test.p_threshold)}"
    else:
        return f"Difference between A1 and A2 frequencies are not significant"
