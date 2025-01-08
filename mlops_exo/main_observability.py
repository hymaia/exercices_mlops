import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, DataQualityPreset
from evidently.metric_preset.metric_preset import AnyMetric
from evidently.metric_preset.metric_preset import MetricPreset
from evidently.metrics import RegressionAbsPercentageErrorPlot
from evidently.metrics import RegressionErrorBiasTable
from evidently.metrics import RegressionErrorDistribution
from evidently.metrics import RegressionErrorNormality
from evidently.metrics import RegressionErrorPlot
from evidently.metrics import RegressionPredictedVsActualPlot
from evidently.metrics import RegressionPredictedVsActualScatter
from evidently.metrics import RegressionQualityMetric
from evidently.metrics import RegressionTopErrorMetric
from evidently.utils.data_preprocessing import DataDefinition
from evidently import ColumnMapping


if __name__ == "__main__":
    # TODO - exercice 4.3.A : créer les jeux de référence et current
    # ------------------------------------------------------------------------------------
    #
    #
    #
    #
    #
    #
    #
    # ------------------------------------------------------------------------------------

    # TODO - exercice 4.3.B : réaliser le ColumnMapping
    # ------------------------------------------------------------------------------------
    # target =
    # prediction =
    # date_col =
    #  numerical_features = []
    # categorical_features = []
    # column_mapping = ColumnMapping(
    #    target=target,
    #    prediction=prediction,
    #    numerical_features=numerical_features,
    #    categorical_features=categorical_features,
    #    datetime=date_col
    # )
    # ------------------------------------------------------------------------------------

    # créer un rapport de Model Drift
    model_drift_report = Report(
        metrics=[
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(options={"render": {"raw_data": True}}),
            RegressionPredictedVsActualPlot(),
            RegressionAbsPercentageErrorPlot(),
            RegressionTopErrorMetric(),
            RegressionErrorBiasTable(options={"render": {"raw_data": True}}),
        ],
    )
    model_drift_report.run(
        reference_data=None,
        current_data=df_current.sample(10000),
        column_mapping=column_mapping,
    )
    model_drift_report.save_html("../reports/model_drift_report.html")

    # Créer un rapport de Data Drift
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=df_reference,
        current_data=df_current,
        column_mapping=column_mapping,
    )
    data_drift_report.save_html("../reports/data_drift_report.html")
