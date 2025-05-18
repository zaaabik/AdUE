from abc import ABC, abstractmethod


class BaseOutputTransform(ABC):
    @abstractmethod
    def __call__(self, output: dict) -> dict:
        """Calculate uncertainty estimation
        Args:
            output (`dict`): model prediction result
        Returns:
            (`dict`) uncertainty estimation for each model and ensemble
        """
        raise NotImplementedError


class MeanTransform(BaseOutputTransform):
    def __init__(self,
                 transform_feature: str,
                 output_name: str,
                 axis=0):
        super().__init__()
        self.output_name = output_name
        self.axis = axis
        self.transform_feature = transform_feature

    def __call__(self, output: dict) -> dict:
        feature = output[self.transform_feature]
        mean_feature = feature.mean(axis=self.axis)
        return {
            self.output_name: mean_feature
        }



