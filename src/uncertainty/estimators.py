from abc import ABC, abstractmethod

import numpy as np


class BaseUncertaintyEstimator(ABC):
    @abstractmethod
    def __call__(self, output: dict) -> dict:
        """Calculate uncertainty estimation
        Args:
            output (`dict`): model prediction result
        Returns:
            (`dict`) uncertainty estimation for each model and ensemble
        """
        raise NotImplementedError


def ensemble_max_prob(ensemble_probabilities: np.ndarray) -> np.ndarray:
    """Calculate uncertainty estimation based on how is close prediction to 1
    Args:
        ensemble_probabilities: np.ndarray[models_count, dataset_size, num_classes] probability each model for each class
    Returns:
        np.ndarray[dataset_size] common uncertainty estimation for ensemble
    """
    assert len(ensemble_probabilities.shape) == 3
    mean_probabilities = np.mean(ensemble_probabilities, axis=0)
    top_probabilities = np.max(mean_probabilities, axis=-1)
    return 1 - top_probabilities


def ensemble_max_prob_with_coefficients(ensemble_probabilities: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Calculate uncertainty estimation based on how is close prediction to 1
    Args:
        coefficients: np.ndarray[models_count] coefficient for calculated weighted mean
        ensemble_probabilities: np.ndarray[models_count, dataset_size, num_classes] probability each model for each class
    Returns:
        np.ndarray[dataset_size] common uncertainty estimation for ensemble
    """
    assert len(ensemble_probabilities.shape) == 3
    print(coefficients.shape)
    assert len(coefficients.shape) == 3 and coefficients.shape[1] == 1 and coefficients.shape[2] == 1
    mean_probabilities = (ensemble_probabilities * coefficients).sum(axis=0)
    top_probabilities = np.max(mean_probabilities, axis=-1)
    return 1 - top_probabilities


def ensemble_std(ensemble_probabilities: np.ndarray) -> np.ndarray:
    """Calculate uncertainty estimation based on ensemble std
    Args:
        ensemble_probabilities: np.ndarray[models_count, dataset_size, num_classes] probability each model for each class
    Returns:
        np.ndarray[dataset_size] common uncertainty estimation for ensemble
    """
    mean_probabilities = np.mean(ensemble_probabilities, axis=0)[None, :, :]
    uncertainty = ((ensemble_probabilities - mean_probabilities) ** 2).mean(0).sum(-1)
    return uncertainty


def max_prob(probabilities: np.ndarray) -> np.ndarray:
    """Calculate uncertainty estimation based on how is close prediction to 1 for each model
    Args:
        probabilities: np.ndarray[models_count, dataset_size, num_classes] probability each model for each class
    Returns:
        np.ndarray[models_count, dataset_size] uncertainty estimation for each model
    """
    top_probabilities = np.max(probabilities, axis=2)
    return 1 - top_probabilities


class MaxProbabilityUncertaintyEstimator(BaseUncertaintyEstimator):
    """Calculate uncertainty estimation based on how is close prediction to 1 for each model."""

    def __call__(self, output: dict) -> dict:
        """Calculate uncertainty estimation based on how is close prediction to 1 for each model
        Args:
            output (`dict`): model prediction result
        Returns:
            (`dict`) uncertainty estimation for each model and ensemble
        """
        results = {
            "ensemble_max_prob": ensemble_max_prob(output["predictions"]),
            "max_prob": max_prob(output["predictions"]),
        }
        return results


class MaxProbabilityWithCoefficientUncertaintyEstimator(BaseUncertaintyEstimator):
    """Calculate uncertainty estimation based on how is close prediction to 1 for each model."""
    def __init__(self, output_name: str,
                 prediction_field: str = 'predictions',
                 coefficients_name: str = None):
        self.coefficients_name = coefficients_name
        self.output_name = output_name
        self.prediction_field = prediction_field

    def __call__(self, output: dict) -> dict:
        """Calculate uncertainty estimation based on how is close prediction to 1 for each model
        Args:
            output (`dict`): model prediction result
        Returns:
            (`dict`) uncertainty estimation for each model and ensemble
        """
        if self.coefficients_name:
            res = ensemble_max_prob_with_coefficients(
                output[self.prediction_field],
                output[self.coefficients_name]
            )
        else:
            res = ensemble_max_prob(
                output[self.prediction_field]
            )

        results = {
            self.output_name: res
        }
        return results


class EnsembleStdUncertaintyEstimator(BaseUncertaintyEstimator):
    """Calculate uncertainty estimation based on how is close prediction to 1 for each model."""

    def __call__(self, output: dict) -> dict:
        """Calculate uncertainty estimation based on how is close prediction to 1 for each model
        Args:
            output (`dict`): model prediction result
        Returns:
            (`dict`) uncertainty estimation for each model and ensemble
        """
        results = {"ensemble_variance": ensemble_std(output["predictions"])}
        return results
