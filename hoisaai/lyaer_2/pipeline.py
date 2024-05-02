import typing
from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel
from hoisaai.lyaer_2.metric import Metric


def pipeline(
    models: typing.Dict[str, SupervisedLearningModel],
    sample_labels: typing.List[str],
    in_samples: typing.List[Tensor],
    out_of_samples: typing.List[Tensor],
    number_of_target: int,
    metric: Metric,
) -> typing.Iterable[Tensor]:
    for destination, model in models.items():
        for sample_index, sample_label in enumerate(sample_labels):
            model.fit(
                in_sample=in_samples[sample_index],
                number_of_target=number_of_target,
            )
            (
                # (..., Out-of-sample observation, Feature)
                out_of_sample_x,
                # (..., Out-of-sample observation, Target)
                out_of_sample_y,
            ) = out_of_samples[sample_index].get_sample_x_and_y(
                number_of_target=number_of_target,
            )
            evaluate: Tensor = metric.evaluate(
                prediction=model.predict(
                    sample_x=out_of_sample_x,
                ),
                sample=out_of_sample_y,
            )
            evaluate.save(
                destination=destination.format(sample_label),
            )
            yield evaluate
