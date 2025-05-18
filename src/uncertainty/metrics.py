from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torchmetrics.classification import MulticlassCalibrationError, BinaryAUROC
from typing import Any


class UEMetric(ABC):
    @abstractmethod
    def __call__(self, estimator: np.ndarray, target: np.ndarray) -> float:
        """Abstract method. Measures the quality of uncertainty estimations `estimator` by
        comparing them to the ground-truth uncertainty estimations `target`.

        Parameters:
            estimator (np.ndarray): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (np.ndarray): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: a quality measure of `estimator` estimations.
                Higher values can indicate either better or lower qualities,
                which depends on a particular implementation.
        """
        raise NotImplementedError("Not implemented")


def normalize(target: np.ndarray):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


@dataclass
class BaseMetricOutput:
    ensemble_metric: np.ndarray | None
    standalone_models_metric: np.ndarray | None


class BaseMetric(ABC):
    """Class for basic behavior of metric calculation.

    If you want to add a new metric, then you need to be inherited from this base class and
    override several functions.
    """

    higher_is_better = None
    metric_name = None

    def __init__(self, prediction_field: str, target_field: str):
        self.prediction_field = prediction_field
        self.target_field = target_field
        if self.metric_name is None or self.higher_is_better is None:
            raise ValueError('You should provide metric and higher_is_better')

    def __str__(self):
        """Return name of model
        Returns:
            (str) name of metric
        """
        if self.higher_is_better:
            arrow = '↑'
        else:
            arrow = '↓'

        return f'{self.metric_name} | {arrow}'

    def _metric_compute(self, predictions: np.ndarray, targets: np.ndarray) -> BaseMetricOutput:
        """Parameters:
            predictions: np.ndarray[num_models, items, classes] probability of model ensemble
            targets: np.ndarray[items] integer number of true class
        Returns:
            BaseMetricOutput metric for each model and ensemble
        """
        raise NotImplementedError

    def __call__(self, outputs: dict) -> BaseMetricOutput:
        """Produce metric for each model and ensemble
        Parameters:
            outputs (dict): model predictions
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        predictions: np.ndarray = outputs[self.prediction_field]
        targets = outputs[self.target_field]
        return self._metric_compute(predictions, targets)


class BaseUEMetric(ABC):
    """Class for basic behavior of metric calculation.

    If you want to add a new metric, then you need to be inherited from this base class and
    override several functions.
    """

    higher_is_better = None
    metric_name = None

    def __init__(self, ue_estimator_field: str, prediction_field: str, target_field: str):
        self.ue_estimator_field = ue_estimator_field
        self.prediction_field = prediction_field
        self.target_field = target_field
        if self.metric_name is None or self.higher_is_better is None:
            raise ValueError('You should provide metric and higher_is_better')

    def __str__(self):
        """Return name of model
        Returns:
            (str) name of metric
        """
        if self.higher_is_better:
            arrow = '↑'
        else:
            arrow = '↓'

        return f'{self.metric_name} | {self.ue_estimator_field} | {arrow}'

    def _metric_compute(self, predictions: np.ndarray, targets: np.ndarray) -> BaseMetricOutput:
        """Parameters:
            predictions: np.ndarray[num_models, items, classes] probability of model ensemble
            targets: np.ndarray[items] integer number of true class
        Returns:
            BaseMetricOutput metric for each model and ensemble
        """
        raise NotImplementedError

    def __call__(self, outputs: dict) -> BaseMetricOutput:
        """Produce metric for each model and ensemble
        Parameters:
            outputs (dict): model predictions
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        predictions: np.ndarray = outputs[self.prediction_field]
        targets = outputs[self.target_field]
        return self._metric_compute(predictions, targets)


class Accuracy(BaseMetric):
    """Accuracy metric."""

    higher_is_better = True
    metric_name = 'accuracy'

    def _metric_compute(self, predictions: np.ndarray, targets: np.ndarray) -> BaseMetricOutput:
        """Parameters:
            predictions: np.ndarray[num_models, items, classes] probability of model ensemble
            targets: np.ndarray[items] integer number of true class
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        assert np.allclose(predictions.sum(axis=2), 1.0)
        original_predictions = predictions
        # if predictions.shape[2] == 2:  # binary classification
        predictions = copy(predictions.argmax(axis=-1))

        num_models = predictions.shape[0]
        standalone_models_metric = []
        for i in range(num_models):
            standalone_models_metric.append(accuracy_score(targets, predictions[i]) * 100.0)

        tp = original_predictions.sum(axis=0)
        ensemble_metric = accuracy_score(targets, tp.argmax(axis=-1)) * 100.0

        return BaseMetricOutput(
            standalone_models_metric=np.array(standalone_models_metric),
            ensemble_metric=np.array(ensemble_metric),
        )


class RocAUCMetric(BaseMetric):
    """RocAuc metric."""

    higher_is_better = True
    metric_name = 'roc-auc'

    def _metric_compute(self, predictions: np.ndarray, targets: np.ndarray) -> BaseMetricOutput:
        """Parameters:
            predictions: (np.ndarray[num_models, items, classes]) probability of model ensemble
            targets: np.ndarray(items) integer number of true class
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        assert np.allclose(predictions.sum(axis=2), 1.0)

        if predictions.shape[2] == 2:  # binary classification
            predictions = predictions[:, :, 1]
        num_models = predictions.shape[0]
        standalone_models_metric = []
        for i in range(num_models):
            standalone_models_metric.append(
                roc_auc_score(targets, predictions[i], multi_class="ovo") * 100.0
            )
        ensemble_metric = (
            roc_auc_score(targets, predictions.mean(axis=0), multi_class="ovo") * 100.0
        )

        return BaseMetricOutput(
            standalone_models_metric=np.array(standalone_models_metric),
            ensemble_metric=np.array(ensemble_metric),
        )


class ExpectedCalibrationError(BaseMetric):
    """Expected calibration error metric."""

    higher_is_better = False
    metric_name = 'Expected calibration error'

    def _metric_compute(self, predictions: np.ndarray, targets: np.ndarray) -> BaseMetricOutput:
        """Parameters:
            predictions: (np.ndarray[num_models, items, classes]) probability of model ensemble
            targets: np.ndarray(items) integer number of true class
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        assert np.allclose(predictions.sum(axis=2), 1.0)
        predictions = torch.tensor(predictions, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)

        num_models = predictions.shape[0]
        standalone_models_metric = []
        num_classes = predictions.shape[-1]

        for i in range(num_models):
            standalone_models_metric.append(
                MulticlassCalibrationError(num_classes=num_classes)(predictions[i], targets)
                * 100.0
            )

        ensemble_metric = (
            MulticlassCalibrationError(num_classes=num_classes)(predictions.mean(axis=0), targets)
            * 100.0
        )

        return BaseMetricOutput(
            standalone_models_metric=np.array(standalone_models_metric),
            ensemble_metric=np.array(ensemble_metric),
        )


class RCAUCMetric(BaseUEMetric):
    """Risk-Coverage curve auc"""

    metric_name = 'Risk Coverage curve auc'
    higher_is_better = False

    def __init__(self, ue_estimator_field: str, prediction_field: str, target_field: str):
        super().__init__(ue_estimator_field, prediction_field, target_field)

    def __call__(self, outputs: dict) -> BaseMetricOutput:
        """Produce metric rejection curve auc for each model and ensemble
        Parameters:
            outputs (dict): model predictions
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        print(outputs.keys())
        predictions = outputs[self.prediction_field].argmax(axis=-1)
        targets: np.ndarray = outputs[self.target_field]

        total_models = predictions.shape[0]
        metric = RiskCoverageCurveAUC()
        true_confidence: np.ndarray
        model_prediction: np.ndarray
        metric_values: np.ndarray | list | None = []

        if self.ue_estimator_field in outputs:
            uncertainty = outputs[self.ue_estimator_field]
            for model_number in range(total_models):
                ue_estimation = uncertainty[model_number]
                model_prediction = predictions[model_number]
                true_confidence = (model_prediction == targets).astype(int)
                rcc_auc = metric(ue_estimation, true_confidence)
                metric_values.append(rcc_auc)
            metric_values = np.array(metric_values)
        else:
            metric_values = None

        ensemble_metric_name = f"ensemble_{self.ue_estimator_field}"
        if ensemble_metric_name in outputs:
            metric = RiskCoverageCurveAUC()
            ensemble_uncertainty = outputs[ensemble_metric_name]
            ensemble_metric_values: np.ndarray | list | None = []
            for model_number in range(total_models):
                model_prediction = predictions[model_number]
                true_confidence = (model_prediction == targets).astype(int)
                rcc_auc = metric(ensemble_uncertainty, true_confidence)
                ensemble_metric_values.append(rcc_auc)
            ensemble_metric_values = np.array(ensemble_metric_values)
        else:
            ensemble_metric_values = None

        return BaseMetricOutput(
            ensemble_metric=ensemble_metric_values,
            standalone_models_metric=metric_values,
        )


class ROCAUCUEMetric(BaseUEMetric):
    """ROC-AUC UE METRIC"""

    metric_name = 'ROC AUC UE'
    higher_is_better = True

    def __init__(self, ue_estimator_field: str, prediction_field: str, target_field: str):
        super().__init__(ue_estimator_field, prediction_field, target_field)

    def __call__(self, outputs: dict) -> BaseMetricOutput:
        """Produce metric rejection curve auc for each model and ensemble
        Parameters:
            outputs (dict): model predictions
        Returns:
            (BaseMetricOutput) metric for each model and ensemble
        """
        assert len(outputs[self.prediction_field].shape) == 3
        assert np.allclose(outputs[self.prediction_field].sum(axis=2), 1.)
        predictions = outputs[self.prediction_field].argmax(axis=-1)
        targets: np.ndarray = outputs[self.target_field]
        errors = targets != predictions

        total_models = predictions.shape[0]
        true_confidence: np.ndarray
        model_prediction: np.ndarray
        metric_values: np.ndarray | list | None = []

        if self.ue_estimator_field in outputs:
            uncertainty = outputs[self.ue_estimator_field]
            for model_number in range(total_models):
                ue_estimation = uncertainty[model_number]
                model_errors = errors[model_number]
                roc_auc = roc_auc_score(model_errors, ue_estimation) * 100.
                metric_values.append(roc_auc)
            metric_values = np.array(metric_values)
        else:
            metric_values = None

        ensemble_metric_name = f"ensemble_{self.ue_estimator_field}"
        if ensemble_metric_name in outputs:
            ensemble_uncertainty = outputs[ensemble_metric_name]
            ensemble_metric_values: np.ndarray | list | None = []
            for model_number in range(total_models):
                model_errors = errors[model_number]
                roc_auc = roc_auc_score(model_errors, ensemble_uncertainty) * 100.
                ensemble_metric_values.append(roc_auc)
            ensemble_metric_values = np.array(ensemble_metric_values)
        else:
            ensemble_metric_values = None

        return BaseMetricOutput(
            ensemble_metric=ensemble_metric_values,
            standalone_models_metric=metric_values,
        )


class RiskCoverageCurveAUC(UEMetric):
    """Calculates area under the Risk-Coverage curve."""

    def __init__(self, normalize_risk_curve: bool = True):
        """Parameters:
        normalize_risk_curve (bool): whether the risk curve should be normalized to 0..1
        """
        super().__init__()
        self.normalize_risk_curve = normalize_risk_curve

    def __call__(self, estimator: np.ndarray, target: np.ndarray) -> float:
        """Measures the area under the Risk-Coverage curve between `estimator` and `target`.

        Parameters:
            estimator (ndarray): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (ndarray): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Risk-Coverage curve.
                Lower values indicate better uncertainty estimations.
        """
        target = normalize(target)
        risk = 1 - np.array(target)
        cr_pair = list(zip(estimator, risk))
        cr_pair.sort(key=lambda x: x[0])
        cumulative_risk = np.cumsum([x[1] for x in cr_pair])
        if self.normalize_risk_curve:
            cumulative_risk = cumulative_risk / np.arange(1, len(estimator) + 1)
        return cumulative_risk.sum()


class MetricSummationCoefficient:
    def __init__(self,
                 metric_name: str,
                 output_name: str,
                 higher_is_better: bool):
        self.higher_is_better = higher_is_better
        self.metric_name = metric_name
        self.output_name = output_name

    def __call__(self, metrics: dict) -> dict[str, np.ndarray]:
        metric_values = metrics[self.metric_name].standalone_models_metric

        if not self.higher_is_better:
            metric_values = 1 / (metric_values + 1e-6)

        normalized_metrics = metric_values / metric_values.sum()

        return {self.output_name: normalized_metrics[:, None, None] * metrics['predictions']}


def max_prob(logits):
    assert logits.ndim == 2
    probs = torch.softmax(logits, dim=1)
    p_max = probs.amax(dim=1)
    return 1 - p_max


class ErrorAUROCMaxProb(BinaryAUROC):
    """ROC-AUC metric for error and max prob"""

    @staticmethod
    def transform(pred, targets) -> (torch.Tensor, torch.Tensor):
        class_preds = pred.argmax(dim=1)
        errors = class_preds != targets
        return max_prob(pred), errors

    def update(self, pred, targets) -> None:
        pred, targets = self.transform(pred, targets)
        super().update(pred, targets)

    def forward(self, pred, targets) -> Any:
        """Wrap the forward call of the underlying metric."""
        return super().forward(pred, targets)
