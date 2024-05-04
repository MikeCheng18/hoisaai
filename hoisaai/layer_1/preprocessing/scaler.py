from hoisaai.layer_0.tensor import Tensor


class Scaler(object):
    def __init__(self) -> None:
        pass

    def fit(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def inverse_transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def transform(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class StandardScaler(Scaler):
    def __init__(self) -> None:
        self.mean: Tensor = None
        self.std: Tensor = None

    def fit(self, x: Tensor) -> Tensor:
        self.mean = x.mean(axis=-2, keep_dimension=True)
        self.std = x.std(axis=-2, keep_dimension=True)
        return (x - self.mean) / self.std

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std
