from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ff3.drift.features import FeatureReport, UnknownFeatureReport, BivariateFeatureReport



class PlotService(ABC):

    @abstractmethod
    def _prepare(self):
        pass 

    @property
    @abstractmethod
    def artifact(self):
        pass


class OverallPlotService(PlotService):
    def __init__(self, dr: 'DriftReport'):
        self.dr = dr
        self.feature_reports = sorted(dr.feature_reports.values(), key=lambda x: x.name)
        self.__message = ""
        self._is_prepared = False

    def __display_overall_drift_probability(self):
        status = "Unknown"
        if 0 <= self.dr.odp < 25:
            status = "Normal"
        elif 25 <= self.dr.odp < 50:
            status = "Low drift detected"
        elif 50 <= self.dr.odp < 75:
            status = "Medium drift detected"
        else:
            status = "Severe drift detected"
        self.__message += f"Overall status: {status}\n"

    def __display_warnings(self):
        warnings = [r.get_warning() for r in self.feature_reports]
        warnings = list(filter(None, warnings))
        if warnings: 
            self.__message += "Warnings:\n"
            for warning in warnings:
                self.__message += f"  {warning.get('message')}\n"

    def _prepare(self):
        self.__display_overall_drift_probability()
        self.__display_warnings()
        self._is_prepared = True

    @property
    def artifact(self):
        if not self._is_prepared: 
            self._prepare()
        return self.__message

class DetailedPlotService(PlotService):
    def __init__(self, dr: 'DriftReport'):
        self.feature_reports = sorted(dr.feature_reports.values(), key=lambda x: x.name)
        assert len(self.feature_reports) > 0, \
            "At least one report should be available for visualization"

        self.__fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}],
                  [{"type": "table"}]],
            row_heights=[6,5],
        )
        self._is_prepared = False
        
    def __add_bars_from_feature_report(self, fr: FeatureReport, row: int, col: int, visibility: bool):
        """
        Add bars from the first and second dataset.
        """
        self.__fig.add_trace(
            go.Bar(
                x=fr.bins,
                y=fr.a1_hist_values,
                name="A1",
                visible=visibility,
            ),
            row=row,col=col,
        )
        self.__fig.add_trace(
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
        self.__fig.add_trace(
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
        self.__fig.update_layout(
            width=800,
            height=900,
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label=self.__get_dropdown_label(feature), 
                             method="update", 
                             args=[{"visible": self.__define_visibility(i, len(self.feature_reports))},
                                   {"title": repr(feature)}])
                        for i, feature in enumerate(self.feature_reports)
                    ])
                )
            ]
        )

    def __update_yaxis(self):
        self.__fig.update_layout(
            yaxis=dict(
                title="Density",
                titlefont_size=16,
                tickfont_size=14,
            )
        )

    def __set_title(self, fr: FeatureReport):
        self.__fig.update_layout(
            title=repr(fr)
        )

    @staticmethod
    def __define_visibility(index, length):
        assert index < length, "Index should be less than length"
        result = [False, False, False] * index  # Since we have 3 traces, we define 3 base elements per view
        result.extend([True, True, True])
        result.extend([False, False, False] * (length - index - 1))
        return result

    def _prepare(self):
        for i, report in enumerate(self.feature_reports):
            vbar, vtable = [i == 0] * 2
            self.__add_bars_from_feature_report(report, row=1, col=1, visibility=vbar)
            self.__add_table_from_feature_report(report, row=2, col=1, visibility=vtable)
        self.__update_yaxis()
        self.__add_dropdown()
        self.__set_title(self.feature_reports[0])
        self._is_prepared = True

    @property
    def artifact(self):
        if not self._is_prepared:
            self._prepare()
        return self.__fig


class BivariatePlotService(PlotService):
    def __init__(self, bfr: BivariateFeatureReport):
        self.bfr = bfr
        self.__fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.2,
            row_heights=[5,5],
        )
        self.x_title = self.bfr.a1_heatmap.x_title
        self.y_title = self.bfr.a1_heatmap.y_title
        self.a1_heatmap = self.bfr.a1_heatmap
        self.a2_heatmap = self.bfr.a2_heatmap
        self._is_prepared = False

    def __add_heatmaps(self):
        self.__fig.add_trace(
            go.Heatmap(
                name="A1",
                z=self.a1_heatmap.intensity,
                x=self.a1_heatmap.x_labels,
                y=self.a1_heatmap.y_labels,
                coloraxis="coloraxis",
                customdata=np.dstack((self.a1_heatmap.intensity, self.a2_heatmap.intensity)),
                hovertemplate=
                self.y_title + ": %{y}<br>" +
                self.x_title + ": %{x}<br><br>" +
                "A1 Density: %{customdata[0]:.3f}<br>" +
                "A2 Density: %{customdata[1]:.3f}<br>",
            ),
            row=1, col=1,
        )
        self.__fig.add_trace(
            go.Heatmap(
                name="A2",
                z=self.a2_heatmap.intensity,
                x=self.a2_heatmap.x_labels,
                y=self.a2_heatmap.y_labels,
                coloraxis="coloraxis",
                customdata=np.dstack((self.a1_heatmap.intensity, self.a2_heatmap.intensity)),
                hovertemplate=
                self.y_title + ": %{y}<br>" +
                self.x_title + ": %{x}<br><br>" +
                "A1 density: %{customdata[0]:.3f}<br>" +
                "A2 density: %{customdata[1]:.3f}<br>",
            ),
            row=2, col=1,
        )
    
    def __update_axes(self):
        self.__fig.update_xaxes(
            row=1, col=1,
            title_text=self.x_title,
        )
        self.__fig.update_xaxes(
            row=2, col=1,
            title_text=self.x_title,
        )
        self.__fig.update_yaxes(
            row=1, col=1,
            title_text=self.y_title,
        )
        self.__fig.update_yaxes(
            row=2, col=1,
            title_text=self.y_title,
        )

    def __set_title(self):
        self.__fig.update_layout(title=repr(self.bfr))
    
    def _prepare(self):
        self.__add_heatmaps()
        self.__update_axes()
        self.__set_title()
        self._is_prepared = True

    @property
    def artifact(self):
        if not self._is_prepared:
            self._prepare()
        return self.__fig
