import tensorflow as tf
from argparse import ArgumentParser

from libs.DataHandler import MNIST
from libs.ExperimentWrapper import ExperimentWrapper

from libs.constants import add_standard_arguments, ALARM_SMALL, ALARM_BIG

# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# Configuration
this_parse = ArgumentParser(description="Train DA3D on MNIST")
add_standard_arguments(this_parse)
this_args = this_parse.parse_args()

experiment_config = [
    MNIST(
        random_state=this_args.random_seed, y_normal=list(range(1, 14)), y_anomalous=list(range(14, 27)),
        n_train_anomalies=this_args.n_train_anomalies, p_pollution=this_args.p_contamination, special_name="emnist"
    ),
]

DIM_TARGET = None
DIM_ALARM = ALARM_BIG
BATCH_SIZE = 512
SAMPLE_DEV = this_args.sample_stddev

if __name__ == '__main__':

    this_experiment = ExperimentWrapper(
        save_prefix="EMNIST", data_setup=experiment_config, p_contamination=this_args.p_contamination,
        random_seed=this_args.random_seed, out_path=this_args.model_path, is_override=this_args.is_override
    )

    dagmm_conf = {
        "comp_hiddens": [60, 30, 10, 1], "comp_activation": tf.nn.tanh,
        "est_hiddens": [10, 4], "est_dropout_ratio": 0.5, "est_activation": tf.nn.tanh,
        "learning_rate": this_args.learning_rate, "epoch_size": this_args.n_epochs, "minibatch_size": BATCH_SIZE
    }

    this_experiment.do_everything(
        dim_target=DIM_TARGET, dim_alarm=DIM_ALARM,
        learning_rate=this_args.learning_rate, batch_size=BATCH_SIZE, n_epochs=this_args.n_epochs,
        out_path=this_args.result_path, dagmm_conf=dagmm_conf, evaluation_split=this_args.data_split,
        stddev_anomalies=SAMPLE_DEV, plot_freq=this_args.plot_freq
    )
