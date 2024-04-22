import typing
import jax
import jaxlib.xla_extension

from hoisaai.layer_0.tensor import split_x_y
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel
from hoisaai.layer_1.model import Tensor, get_tensor


class KNearestNeighbors(SupervisedLearningModel):

    def __init__(
        self,
        k: int,
        tensor: Tensor = None,
        number_of_dependent_variables: int = None,
    ) -> None:
        super().__init__(
            tensor=tensor,
            number_of_dependent_variables=number_of_dependent_variables,
        )
        self.k: int = k

    def __str__(self) -> str:
        return f"K Nearest Neighbors: k={self.k:.3f}"

    def _predict(
        self,
        in_sample: jaxlib.xla_extension.ArrayImpl,
        out_of_sample: jaxlib.xla_extension.ArrayImpl,
    ) -> jaxlib.xla_extension.ArrayImpl:
        (
            # (..., In-sample observation, Independent variable)
            in_sample_x,
            # (..., In-sample observation, Dependent variable)
            in_sample_y,
        ) = split_x_y(
            tensor=in_sample,
            number_of_dependent_variables=self.number_of_dependent_variables,
        )
        (
            # (..., Out-of-sample observation, Independent variable)
            out_of_sample_x,
            # (..., Out-of-sample observation, Dependent variable)
            out_of_sample_y,
        ) = split_x_y(
            tensor=out_of_sample,
            number_of_dependent_variables=self.number_of_dependent_variables,
        )
        unknown_dimension: typing.Tuple[int] = out_of_sample_y.shape[:-2]
        # (..., Out-of-sample observation, Dependent variable)
        return (
            # (..., Out-of-sample observation, Dependent variable)
            jax.numpy.average(
                # (k, ..., Out-of-sample observation, Dependent variable)
                (
                    # (..., In-sample observation, Dependent variable)
                    in_sample_y
                )[
                    # ...
                    *[
                        list(range(i))
                        * (
                            (
                                self.k
                                * (
                                    int(
                                        jax.numpy.prod(
                                            jax.numpy.array(list(unknown_dimension))
                                        )
                                    )
                                )
                                * out_of_sample.shape[-2]
                            )
                            // i
                        )
                        for i in unknown_dimension
                    ],
                    # (k * ... * Out-of-sample observation)
                    (
                        # (k, ..., Out-of-sample observation)
                        jax.numpy.argsort(
                            # (In-sample observation, ..., Out-of-sample observation)
                            jax.numpy.sqrt(
                                # (In-sample observation, ..., Out-of-sample observation)
                                jax.numpy.sum(
                                    # (In-sample observation, ..., Out-of-sample observation, Independent variable)
                                    jax.numpy.square(
                                        # (In-sample observation, ..., Out-of-sample observation, Independent variable)
                                        (
                                            # (1, ..., Out-of-sample observation, Independent variable)
                                            jax.numpy.expand_dims(
                                                # (..., Out-of-sample observation, Independent variable)
                                                a=out_of_sample_x,
                                                axis=0,
                                            )
                                        )
                                        - (
                                            # (In-sample observation, ..., Out-of-sample observation, Independent variable)
                                            jax.numpy.swapaxes(
                                                # (Out-of-sample observation, ..., In-sample observation, Independent variable)
                                                a=jax.numpy.repeat(
                                                    a=in_sample_x,
                                                    # Out-of-sample observation
                                                    repeats=out_of_sample_x.shape[-2],
                                                    axis=0,
                                                ).reshape(
                                                    # Out-of-sample observation
                                                    out_of_sample_x.shape[-2],
                                                    *in_sample_x.shape,
                                                ),
                                                # Out-of-sample observation
                                                axis1=0,
                                                # In-sample observation
                                                axis2=-2,
                                            )
                                        )
                                    ),
                                    # Independent variable
                                    axis=-1,
                                )
                            )
                        )[
                            # In-sample observation
                            : self.k,
                            ...,
                        ]
                    ).flatten(),
                    # Dependent variable
                    :,
                ].reshape(
                    # (k, ..., Out-of-sample observation, Dependent variable)
                    self.k,
                    *unknown_dimension,
                    out_of_sample_y.shape[-2],
                    in_sample_y.shape[-1],
                ),
                # k
                axis=0,
            )
        )

    def predict(
        self,
        tensor: Tensor,
    ) -> typing.Any:
        return self._predict(
            in_sample=get_tensor(tensor=self.tensor),
            out_of_sample=get_tensor(tensor=tensor),
        )

    def shapley_value(
        self,
        tensor: Tensor,
    ) -> jaxlib.xla_extension.ArrayImpl:
        # (..., In-sample observation, Dependent variable and independent variable)
        in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=self.tensor,
        )
        # (..., Out-of-sample observation, Dependent variable and independent variable)
        out_of_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=tensor,
        )
        number_of_independent_variables: int = (
            out_of_sample.shape[-1] - self.number_of_dependent_variables
        )
        # (..., Out-of-sample observation, Dependent variables, Independent variable)
        shapley_value: jaxlib.xla_extension.ArrayImpl = jax.numpy.full(
            shape=(
                *out_of_sample.shape[:-1],
                self.number_of_dependent_variables,
                number_of_independent_variables,
            ),
            fill_value=jax.numpy.nan,
        )
        # (..., Out-of-sample observation, Dependent variable)
        predicted_value: jaxlib.xla_extension.ArrayImpl = self._predict(
            in_sample=in_sample,
            out_of_sample=out_of_sample,
        )
        for independent_index in range(number_of_independent_variables):
            shapley_value = shapley_value.at[
                ...,
                independent_index,
            ].set(
                predicted_value
                - self._predict(
                    in_sample=jax.numpy.delete(
                        arr=in_sample,
                        obj=independent_index + self.number_of_dependent_variables,
                        axis=-1,
                    ),
                    out_of_sample=jax.numpy.delete(
                        arr=out_of_sample,
                        obj=independent_index + self.number_of_dependent_variables,
                        axis=-1,
                    ),
                )
            )
        return shapley_value

    def transform(self) -> typing.Iterator[typing.Any]:
        yield get_tensor(
            tensor=self.tensor,
        )
