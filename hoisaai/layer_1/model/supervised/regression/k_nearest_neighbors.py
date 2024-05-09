"""K Nearest Neighbors Regressor."""
from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.core.k_nearest_neighbors import (
    KNearestNeighbors,
)
from hoisaai.layer_1.model.supervised.regression.regression import Regression


class KNearestNeighborsRegressor(
    KNearestNeighbors,
    Regression,
):
    """K Nearest Neighbors Regressor.
    
    :param k: Number of nearest neighbors.
    :type k: int
    """
    def __init__(
        self,
        k: int,
    ) -> None:
        KNearestNeighbors.__init__(
            self,
            k=k,
        )
        Regression.__init__(
            self,
        )

    def __str__(self) -> str:
        return "K Nearest Neighbors Regressor: " + f"k={self.k}"

    def predict(
        self,
        # (..., Sample observation, Feature)
        sample_x: Tensor,
    ) -> Tensor:
        # (..., Sample observation, Target)
        return KNearestNeighbors.predict(
            self,
            sample_x=sample_x,
        ).mean(
            # k
            axis=-1,
            keep_dimension=False,
        )
