from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ff3.drift.features import FeatureReport, UnknownFeatureReport



class PlotService(ABC):

    @abstractmethod
    def show(self):
        pass


class OverallPlotService(PlotService):
    def __init__(self, dr: 'DriftReport'):
        self.dr = dr
        self.reports = sorted(dr.feature_reports, key=lambda x: x.name)

    def __display_overall_drift_probability(self):
        print("Overall status: ", end="")
        proba = self.dr.overall_drift_probability()
        if 0 <= proba < 25:
            print("Normal")
        elif 25 <= proba < 50:
            print("Low drift detected")
        elif 50 <= proba < 75:
            print("Medium drift detected")
        else:
            print("Severe drift detected")

    def __display_warnings(self):
        warnings = [r.get_warning() for r in self.reports]
        warnings = list(filter(None, warnings))
        print("Warnings:")
        for warning in warnings:
            print(" "*2, warning.get("message"))
    
    def show(self):
        self.__display_overall_drift_probability()
        self.__display_warnings()


class DetailedPlotService(PlotService):
    def __init__(self, dr: 'DriftReport'):
        self.reports = sorted(dr.feature_reports, key=lambda x: x.name)
        assert len(self.reports) > 0, "At least one report should be available for visualization"

        self.fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}],
                  [{"type": "table"}]],
            row_heights=[10,5],
        )

        for i, report in enumerate(self.reports):
            vbar, vtable = [i == 0] * 2
            self.__add_bars_from_feature_report(report, row=1, col=1, visibility=vbar)
            self.__add_table_from_feature_report(report, row=2, col=1, visibility=vtable)
        self.__update_yaxis()
        self.__add_dropdown()
        self.__set_title(self.reports[0])
        
    def __add_bars_from_feature_report(self, fr: FeatureReport, row: int, col: int, visibility: bool):
        """
        Add bars from the first and second dataset.
        """
        self.fig.add_trace(
            go.Bar(
                x=fr.bins,
                y=fr.a1_hist_values,
                name="A1",
                visible=visibility,
            ),
            row=row,col=col,
        )
        self.fig.add_trace(
            go.Bar(
                x=fr.bins,
                y=fr.a2_hist_values,
                name="A2",
                visible=visibility,
            ),
            row=row,col=col,
        )

    def __add_table_from_feature_report(self, fr: FeatureReport, row: int, col: int, visibility: bool):
        """
        Add table, describing computed characteristics from the feature report
        """
        values = [[
            t.name, 
            t.a1_statistic, 
            t.a2_statistic, 
            f"<b>{t.message}" if t.has_changed else t.message
        ] for t in fr.tests]
        values_transposed = list(zip(*values))

        font_color = ['#2a3f5f', '#2a3f5f', '#2a3f5f']
        font_color.append(['#b2070f' if t.has_changed else '#2a3f5f' for t in fr.tests])
        self.fig.add_trace(
            go.Table(
                header=dict(
                    values=["Name", "A1", "A2", "Change's status"],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=values_transposed,
                    font=dict(color=font_color),
                    align="left"
                ),
                visible=visibility,
            ),
            row=row,col=col,
        )

    def __get_dropdown_label(self, fr: FeatureReport) -> str:
        name = f"⚠️ {fr.name}" if fr.is_drifted else fr.name
        name = f"<i>{name}</i>" if fr.has_failed_tests else name
        return name
    
    def __add_dropdown(self):
        self.fig.update_layout(
            width=800,
            height=900,
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label=self.__get_dropdown_label(feature), 
                             method="update", 
                             args=[{"visible": self.__define_visibility(i, len(self.reports))},
                                   {"title": repr(feature)}])
                        for i, feature in enumerate(self.reports)
                    ])
                )
            ]
        )

    def __update_yaxis(self):
        self.fig.update_layout(
            yaxis=dict(
                title="Density",
                titlefont_size=16,
                tickfont_size=14,
            )
        )

    def __set_title(self, fr: FeatureReport):
        self.fig.update_layout(
            title=repr(fr)
        )

    @staticmethod
    def __define_visibility(index, length):
        assert index < length, "Index should be less than length"
        result = [False, False, False] * index  # Since we have 3 traces, we define 3 base elements per view
        result.extend([True, True, True])
        result.extend([False, False, False] * (length - index - 1))
        return result

    def show(self):
        self.fig.show()
