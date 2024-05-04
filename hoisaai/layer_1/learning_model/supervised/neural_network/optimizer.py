import typing

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.neural_network.module import Parameter


class Optimizer(object):
    def __init__(
        self,
        parameter: typing.Iterator[Parameter],
    ) -> None:
        self.parameter: typing.List[Parameter] = list(parameter)

    def initialize_gradient(self) -> None:
        for parameter in self.parameter:
            parameter.detach()

    def step(self) -> None:
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        parameter: typing.Iterator[Parameter],
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> None:
        Optimizer.__init__(self, parameter)
        self.learning_rate: float = learning_rate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.time: int = 0
        self.velocity: typing.List[Tensor] = [
            Tensor.full(shape=parameter.shape, value=0.0, datatype=parameter.datatype)
            for parameter in self.parameter
        ]
        self.momentum: typing.List[Tensor] = [
            Tensor.full(shape=parameter.shape, value=0.0, datatype=parameter.datatype)
            for parameter in self.parameter
        ]

    def step(self) -> None:
        self.time += 1
        for index, parameter in enumerate(self.parameter):
            velocity: Tensor = self.velocity[index]
            momentum: Tensor = self.momentum[index]
            momentum = self.beta1 * momentum + (1 - self.beta1) * parameter.gradient
            velocity = (
                self.beta2 * velocity + (1 - self.beta2) * parameter.gradient.square()
            )
            momentum_hat: Tensor = momentum / (1 - self.beta1**self.time)
            velocity_hat: Tensor = velocity / (1 - self.beta2**self.time)
            self.parameter[index][...] = parameter - (
                self.learning_rate * momentum_hat / (velocity_hat.sqrt() + self.epsilon)
            )
            self.velocity[index] = velocity
            self.momentum[index] = momentum


class SGD(Optimizer):
    def __init__(
        self,
        parameter: typing.Iterator[Parameter],
        learning_rate: float,
        momentum: float,
    ) -> None:
        Optimizer.__init__(self, parameter)
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.velocity: typing.List[Tensor] = [
            Tensor.full(shape=parameter.shape, value=0.0, datatype=parameter.datatype)
            for parameter in self.parameter
        ]

    def step(self) -> None:
        for index, parameter in enumerate(self.parameter):
            velocity: Tensor = self.velocity[index]
            velocity = (
                velocity * self.momentum - self.learning_rate * parameter.gradient
            )
            self.parameter[index][...] = parameter + velocity
            self.velocity[index] = velocity
