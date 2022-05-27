import random
import re

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Union, List, Tuple, Dict, NoReturn
from pathlib import Path
from warnings import warn
from copy import deepcopy

from libs.DA3D import DA3D
from libs.DataHandler import DataLabels
from libs.Metrics import evaluate_roc, roc_to_pandas
from libs.architecture.target import Autoencoder, AdversarialAutoencoder
from libs.constants import BASE_PATH

from sklearn.metrics import roc_curve, roc_auc_score


class ExperimentWrapper:
    def __init__(
            self, data_setup: List[DataLabels], p_contamination: float = 0.0,
            random_seed: int = None, is_override=False, auto_split: bool = True,
            save_prefix: str = '', out_path: Path = BASE_PATH, auto_subfolder: bool = True,
    ):
        """
        Wrapper class to have a common scheme for the experiments
        :param data_setup: data configuration for every experiment
        :param p_contamination: fraction of contamination, i.e. anomaly samples in the training data
        :param save_prefix: prefix for saved NN models
        :param random_seed: seed to fix the randomness
        :param is_override: override output if it already exists
        :param auto_split: split the data using k-means if only one class if available
        :param out_path: output base path for the models, usually the base path
        :param auto_subfolder: create a subfolder on the out_path with the random seed and the contamination level
        """

        # Save the parameter grid
        self.data_setup = data_setup  # This is mutable to allow target splitting
        self.p_contamination = p_contamination

        # Configuration
        self.is_override = is_override
        self.auto_split = auto_split

        # Folder paths
        self.out_path = out_path
        if auto_subfolder:
            self.out_path /= f"{p_contamination}_{random_seed}"
        # If necessary, create the output folder
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_prefix = f"{save_prefix}"
        if random_seed is not None:
            self.save_prefix += f"_{random_seed}"

        # Fix randomness
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        # Alright, we can't make the NN deterministic on a GPU [1]. Probably makes more sense to keep the sample
        # selection deterministic, but repeat all NN-related aspects.
        # [1] https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        # tf.random.set_seed(random_seed)

    def evaluate_ad(
            self, data_split: str,
            architecture_params: dict = None, target_params: dict = None,
            evaluate_baseline: Dict[str, dict] = None, out_path: Path = None
    ) -> NoReturn:
        """
        Evaluate the performance of DA3D
        :param data_split: data split, e.g. val or test
        :param architecture_params: DA3D architecture settings
        :param target_params: target network's architecture settings
        :param evaluate_baseline: also evaluate the given baseline methods, expects a dict of {"baseline": {config}}
        :param out_path: special output path for the results
        :return:
        """

        if architecture_params is None: architecture_params = {}
        if target_params is None: target_params = {}
        if out_path is None: out_path = self.out_path

        for i_data, cur_data in enumerate(deepcopy(self.data_setup)):
            # Start with a new session
            plt.clf()
            tf.keras.backend.clear_session()

            # We'll output the metrics and the x,y coordinates for the ROC
            df_metric = pd.DataFrame(columns=["AUC", "AP"])
            df_roc = pd.DataFrame()

            # Announce what we're doing
            component_prefix = self.parse_name(cur_data, is_supervised=False)
            ad_prefix = self.parse_name(cur_data, is_supervised=True)
            print(f"Now evaluating {ad_prefix}")

            # Get the output path
            # Check if the respective model exists
            csv_path = self.get_model_path(
                base_path=out_path, file_name=ad_prefix,
                file_suffix=".csv"
            )

            # Evaluate baseline methods
            if evaluate_baseline:
                for baseline_name, baseline_config in evaluate_baseline.items():
                    print(f"Evaluating {baseline_name}")
                    tf.keras.backend.clear_session()
                    baseline_metric, baseline_roc = self.evaluate_baseline_on(
                        data_split=data_split, baseline=baseline_name, input_config=cur_data, **baseline_config
                    )
                    df_metric.loc[baseline_name, :] = baseline_metric
                    df_roc = pd.concat([df_roc, baseline_roc], axis=1, ignore_index=False)

            # Save the resulting DFs
            df_metric.to_csv(csv_path.with_suffix(".metric.csv"))
            df_roc.to_csv(csv_path.with_suffix(".roc.csv"))

            # Plot the ROC
            plt.plot([0, 1], [0, 1], label="Random Classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            # plt.show()
            # For some reason, mpl tries to use the LaTeX processor when adding "GANomaly" - this might fail
            try:
                plt.savefig(csv_path.with_suffix(".roc.png"))
            except RuntimeError:
                # LaTeX used, but not available - this should not interfere with the rest of the evaluation
                pass

    # -- Baselines --
    @staticmethod
    def _get_baseline_info(baseline: str) -> Tuple[str, bool]:
        """
        Get the right file suffix for the respective baseline
        :param baseline: baseline name
        :return: file suffix and if the method is supervised
        """
        if baseline == "DAGMM":
            file_suffix = ""
            is_supervised = False
        elif baseline == "FenceGAN":
            file_suffix = ".h5"
            is_supervised = False
        elif baseline in [
            "DA3D", "SimpleDA3D", "TrivialDA3D", "TrivialSimpleDA3D", "DA3D-1", "DA3D-2", "DA3D-3", "DA3D-4",
            "AE", "AAE", "DeepSVDD-AE", "A3", "DeepSVDD", "GANomaly", "REPEN", "fAnoGAN"
        ]:
            file_suffix = ".tf"
            is_supervised = False
        else:
            raise NotImplementedError(f"{baseline} is not a known baseline method")

        return file_suffix, is_supervised

    def train_baseline(
            self, baseline: str,
            compile_params: dict = None, fit_params: dict = None,
            stddev_anomaly: float = None, **model_params
    ) -> NoReturn:
        """
        Train some baseline methods
        :param baseline: which baseline method to evaluate
        :param compile_params: arguments for the (optional) compile function
        :param fit_params: arguments for the (optional) fit function
        :param stddev_anomaly: stddev for simple generated anomalies
        :param model_params: extra arguments for the baseline method constructor
        :return:
        """

        # Check if baseline method exists
        file_suffix, is_supervised = self._get_baseline_info(baseline)

        # Default to empty dictionaries
        if compile_params is None: compile_params = {}
        if fit_params is None: fit_params = {}

        for cur_data in self.data_setup:
            # Work on a copy as we might change some parts
            _cur_data = deepcopy(cur_data)

            # Unsupervised methods don't know the training anomalies
            this_prefix = self.parse_name(_cur_data, is_supervised=is_supervised)
            print(f"Now training baseline method '{baseline}' for {this_prefix}")

            # Check if the respective model exists
            out_path = self.get_model_path(
                base_path=self.out_path, file_name=this_prefix,
                file_suffix=file_suffix, sub_folder=baseline
            )
            if not self.is_override and (
                    out_path.exists()
                    or out_path.with_suffix(".overall.h5").exists()
                    or out_path.with_suffix(".tf.index").exists()
            ):
                print("This baseline method was already trained. Use is_override=True to override it.")
                continue

            # Create the parent folder if not existing
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=False)

            # Add generated anomalies if applicable
            if stddev_anomaly is not None:
                # Load the AAE
                this_aae = tf.keras.models.load_model(out_path.parent.parent / "AAE" / out_path.name)
                _cur_data.add_generated_train_anomalies(
                    m_anomaly=this_aae, stddev_anomaly=stddev_anomaly
                )

            # Open the data
            this_data = _cur_data.to_data(for_experts=False)

            # Fit the baseline method
            if baseline in [
                "DA3D", "SimpleDA3D", "TrivialDA3D", "TrivialSimpleDA3D",
                "DA3D-1", "DA3D-2", "DA3D-3", "DA3D-4"
            ]:
                # Load the target autoencoder
                this_ae = tf.keras.models.load_model(out_path.parent.parent / "AAE" / out_path.name)

                # Create DA3D
                # In the following: a hacky way to integrate multiple standard deviations
                if "-" in baseline:
                    # Use the stddev given in the name
                    _model_params = {
                        cur_key: cur_val for cur_key, cur_val in model_params.items() if cur_key != "sample_stddev"
                    }
                    baseline_model = DA3D(
                        m_target=this_ae, sample_stddev=int(baseline[-1]), **_model_params, name=self.save_prefix
                    )
                else:
                    baseline_model = DA3D(m_target=this_ae, **model_params, name=self.save_prefix)
                baseline_model.compile(**compile_params)

                # For the ablation study, we deactivate parts of the architecture
                if baseline == "SimpleDA3D":
                    fit_conf = {"w_generated": 0.0, "w_trivial": 0.0, "w_simple": 1.0}
                elif baseline == "TrivialDA3D":
                    fit_conf = {"w_generated": 0.0, "w_trivial": 1.0, "w_simple": 0.0}
                elif baseline == "TrivialSimpleDA3D":
                    fit_conf = {"w_generated": 0.0, "w_trivial": 1.0, "w_simple": 1.0}
                else:
                    # Otherwise, use the default config
                    fit_conf = {}

                # Fit and save
                baseline_model.fit(
                    x=this_data.train_alarm[0], y=this_data.train_alarm[1],
                    validation_data=this_data.val_alarm, **fit_conf, **fit_params
                )

                baseline_model.save(out_path)

            elif baseline == "fAnoGAN":
                from baselines.anogan.AnoGAN import fAnoGAN

                # Load the AE
                this_ae = tf.keras.models.load_model(out_path.parent.parent / "AAE" / out_path.name)
                # Create the baseline
                baseline_model = fAnoGAN(base_ae=this_ae, **model_params)

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.compile(**compile_params)
                baseline_model.fit(
                    x=this_data.train_target[0],
                    **fit_params
                )

                # Save the baseline
                baseline_model.save_weights(out_path)

            elif baseline == "REPEN":
                from baselines.repen.REPEN import REPEN

                baseline_model = REPEN(**model_params)
                # Call once to build the model
                baseline_model(this_data.val_alarm[0].reshape(this_data.val_alarm[0].shape[0], -1))
                baseline_model.compile(**compile_params)

                # REPEN is an outlier detection method - it expects to find some anomalies in the training data
                # If no pollution was applied, give REPEN the advantage of working on the test data
                repen_data = this_data.train_alarm if self.p_contamination else this_data.test_alarm

                # Fit and save
                baseline_model.fit(
                    x=repen_data[0].reshape(repen_data[0].shape[0], -1),
                    y=repen_data[1].reshape(repen_data[1].shape[0], -1),
                    **fit_params
                )

                baseline_model.save_weights(out_path)

            elif baseline == 'DAGMM':
                from baselines.dagmm_v2 import DAGMM

                # Revert to TF1 for compatibility
                tf.compat.v1.disable_v2_behavior()

                baseline_model = DAGMM(random_seed=self.random_seed, **model_params)
                baseline_model.fit(
                    this_data.train_target[0].reshape(this_data.train_target[0].shape[0], -1), **fit_params
                )

                baseline_model.save(out_path)

                # We need to restore the TF2 behaviour afterwards
                tf.compat.v1.reset_default_graph()
                tf.compat.v1.enable_v2_behavior()

            elif baseline in ["AE", "AAE", "DeepSVDD-AE"]:
                # Use the right architecture
                if baseline == "AE":
                    this_net = Autoencoder(**model_params)
                elif baseline == "AAE":
                    this_net = AdversarialAutoencoder(**model_params)
                elif baseline == "DeepSVDD-AE":
                    from baselines.deep_svdd.DeepSVDD import DeepSVDDAE
                    this_net = DeepSVDDAE(**model_params)
                else:
                    raise NotImplementedError("Unknown AE architecture")

                if "optimizer" in compile_params:
                    raise NotImplementedError("Please omit the 'optimizer' keyword. So far only Adam is supported. Specify the LR directly.")
                this_net.compile(**compile_params)

                this_net.fit(
                    x=this_data.train_target[0], y=this_data.train_target[1],
                    validation_data=this_data.val_target,
                    **fit_params
                )

                this_net.save(out_path)

            elif baseline in ["DeepSVDD"]:
                from baselines.deep_svdd.DeepSVDD import DeepSVDD

                # Load the DeepSVDD-AE
                this_ae = tf.keras.models.load_model(out_path.parent.parent / "DeepSVDD-AE" / out_path.name)

                # Create the baseline
                if baseline == "DeepSVDD":
                    baseline_model = DeepSVDD(pretrained_ae=this_ae, **model_params)
                else:
                    raise NotImplementedError

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.calculate_c(this_data.train_target[0])
                baseline_model.compile(**compile_params)
                baseline_model.fit(
                    x=this_data.train_target[0],
                    **fit_params
                )

                # Save the baseline
                baseline_model.save_weights(out_path)

            elif baseline in ["A3"]:
                from baselines.a3.A3v2 import A3

                # Load the target autoencoder
                this_ae = tf.keras.models.load_model(out_path.parent.parent / "AE" / out_path.name)

                # Create A3
                baseline_model = A3(m_target=this_ae, **model_params)
                baseline_model.compile(**compile_params)

                # Fit and save
                baseline_model.fit(
                    x=this_data.train_alarm[0], y=this_data.train_alarm[1],
                    validation_data=this_data.val_alarm, **fit_params
                )

                baseline_model.save_weights(out_path)

            elif baseline == "GANomaly":
                from baselines.tf2_ganomaly.model import GANomaly, opt, batch_resize

                # Check if the input data contains images as we need to scale them
                if len(this_data.data_shape) > 1:
                    x_train = batch_resize(this_data.train_target[0], (32, 32))[..., None]
                    x_val = batch_resize(this_data.val_alarm[0], (32, 32))[..., None]
                    opt.isize = 32
                else:
                    x_train = this_data.train_target[0]
                    x_val = this_data.val_alarm[0]
                    opt.isize = x_train.shape[-1]

                # Use their option object
                opt.encdims = model_params["enc_dims"]

                # Convert data
                # Although GANomaly is unsupervised, it needs training labels: just use zeros
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_train, np.zeros((this_data.train_target[0].shape[0], )))
                )
                val_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_val, this_data.val_alarm[1])
                )
                train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(opt.batch_size, drop_remainder=True)
                val_dataset = val_dataset.batch(opt.batch_size, drop_remainder=False)

                # Construct and train
                baseline_model = GANomaly(opt, train_dataset=train_dataset)
                baseline_model.fit(opt.niter)

                # Save
                baseline_model.save(out_path)

            elif baseline == "FenceGAN":
                from baselines.fencegan_v2.fgan import FenceGAN

                # Fence GAN expects data in [-1, 1]
                x_train = (this_data.train_target[0] - .5) * 2

                data_set = tf.data.Dataset.from_tensor_slices(
                    (x_train, np.zeros((x_train.shape[0], 1)))
                )

                # Construct and train
                baseline_model = FenceGAN(**model_params)
                baseline_model.compile(**compile_params)
                baseline_model.fit(
                    data_set, **fit_params
                )

                # Save
                baseline_model.save(out_path)

    def evaluate_baseline_on(
            self, data_split: str, baseline: str, input_config, **model_params
    ) -> Tuple[list, pd.DataFrame]:
        """
        Evaluate a baseline method on a given data config
        :param data_split: data split, e.g. val or test
        :param baseline: which baseline method to evaluate
        :param input_config: configuration the baseline is evaluated on (takes test data)
        :param model_params: extra arguments for the baseline method constructor
        :return: DataFrame containing the metrics & DataFrame containing the ROC x,y data
        """

        # Check if baseline method exists
        file_suffix, is_supervised = self._get_baseline_info(baseline)

        this_prefix = self.parse_name(input_config, is_supervised=is_supervised)

        # Handle the file origins
        in_path = self.get_model_path(
            base_path=self.out_path, file_name=this_prefix,
            file_suffix=file_suffix, sub_folder=baseline
        )

        # Open the baseline and predict
        this_data = input_config.to_data(test_type=data_split, for_experts=False)
        pred_y = None
        try:
            if baseline in [
                "DA3D", "SimpleDA3D", "TrivialDA3D", "TrivialSimpleDA3D",
                "DA3D-1", "DA3D-2", "DA3D-3", "DA3D-4"
            ]:

                # Load DA3D
                baseline_model = tf.keras.models.load_model(in_path)

                pred_y = baseline_model.predict(x=this_data.test_alarm[0])

            elif baseline == "fAnoGAN":
                from baselines.anogan.AnoGAN import fAnoGAN

                # Load the AE
                this_ae = tf.keras.models.load_model(in_path.parent.parent / "AAE" / in_path.name)
                # Create the baseline
                baseline_model = fAnoGAN(base_ae=this_ae, **model_params)
                # Load the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)

                # Get the variables for the anomaly score
                x_gen, f_in, f_gen = baseline_model(this_data.test_alarm[0])
                pred_y = baseline_model.anomaly_score(
                    x_in=this_data.test_alarm[0], x_gen=x_gen,
                    f_in=f_in, f_gen=f_gen
                )

            elif baseline == "REPEN":
                from baselines.repen.REPEN import REPEN

                # Load REPEN
                baseline_model = REPEN(**model_params)
                # Call once to build the model
                baseline_model(this_data.val_alarm[0].reshape(this_data.val_alarm[0].shape[0], -1))
                baseline_model.load_weights(in_path)

                pred_y = baseline_model.predict(
                    x=this_data.test_alarm[0].reshape(this_data.test_alarm[0].shape[0], -1)
                )

            elif baseline == 'DAGMM':
                from baselines.dagmm_v2 import DAGMM

                baseline_model = DAGMM(random_seed=self.random_seed, **model_params)
                baseline_model.restore(in_path)

                pred_y = baseline_model.predict(
                    this_data.test_alarm[0].reshape((this_data.test_alarm[0].shape[0], -1))
                )

            elif baseline in ["AE", "AAE"]:

                baseline_model = tf.keras.models.load_model(in_path)

                pred_y = baseline_model.m_dec.predict(
                    baseline_model.m_enc(this_data.test_alarm[0])
                )

                # We'll return the MSE as score
                pred_y = np.square(pred_y - this_data.test_alarm[0])
                # We might have 2D inputs: collapse to one dimension
                pred_y = np.reshape(pred_y, (pred_y.shape[0], -1))
                pred_y = np.mean(pred_y, axis=1)

            elif baseline in ["DeepSVDD"]:
                from baselines.deep_svdd.DeepSVDD import DeepSVDD

                # Load the DeepSVDD-AE
                this_ae = tf.keras.models.load_model(in_path.parent.parent / "DeepSVDD-AE" / in_path.name)

                # Create the baseline
                if baseline == "DeepSVDD":
                    baseline_model = DeepSVDD(pretrained_ae=this_ae, **model_params)
                else:
                    raise NotImplementedError

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)
                # c isn't saved, thus we need to recalculate is based on the training data
                baseline_model.calculate_c(this_data.train_target[0])

                # Predict
                pred_y = baseline_model.score(this_data.test_alarm[0])

            elif baseline in ["A3"]:
                from baselines.a3.A3v2 import A3

                # Load the target autoencoder
                this_ae = tf.keras.models.load_model(in_path.parent.parent / "AE" / in_path.name)

                # Load A3
                baseline_model = A3(m_target=this_ae, **model_params)
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)

                pred_y = baseline_model.predict(x=this_data.test_alarm[0])

            elif baseline == "GANomaly":
                from baselines.tf2_ganomaly.model import GANomaly, opt, batch_resize

                # Check if the input data contains images as we need to scale them
                if len(this_data.data_shape) > 1:
                    x_train = batch_resize(this_data.train_alarm[0], (32, 32))[..., None]
                    x_test = batch_resize(this_data.test_alarm[0], (32, 32))[..., None]
                    opt.isize = 32
                else:
                    x_train = this_data.train_alarm[0]
                    x_test = this_data.test_alarm[0]
                    opt.isize = x_train.shape[-1]

                # Use their option object
                opt.encdims = model_params["enc_dims"]

                # Convert data
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_train, this_data.train_alarm[1])
                )
                test_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_test, this_data.test_alarm[1])
                )
                train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(opt.batch_size, drop_remainder=True)
                test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)

                # Construct and predict
                baseline_model = GANomaly(opt, train_dataset=train_dataset, test_dataset=test_dataset)
                baseline_model.load(in_path)

                pred_y = baseline_model._evaluate(test_dataset)[0]

            elif baseline == "FenceGAN":
                from baselines.fencegan_v2.fgan import FenceGAN

                # Fence GAN expects data in [-1, 1]
                x_test = (this_data.test_alarm[0] - .5) * 2

                baseline_model = FenceGAN(**model_params)
                baseline_model.load(in_path)

                pred_y = baseline_model.D.predict(x=x_test)
                # They use 0 for anomalous
                pred_y = 1 - pred_y

            else:
                raise NotImplementedError("Unknown baseline method")

        except FileNotFoundError:
            print(f"No model for {baseline} found. Aborting.")
            return [None, None], pd.DataFrame()

        # Plot ROC
        fpr, tpr, thresholds = roc_curve(
            y_true=this_data.test_alarm[1], y_score=pred_y
        )
        plt.plot(fpr, tpr, label=baseline)

        # Generate the output DF
        df_metric = evaluate_roc(pred_scores=pred_y, test_alarm=this_data.test_alarm)
        df_roc = roc_to_pandas(fpr=fpr, tpr=tpr, suffix=baseline)

        return df_metric, df_roc

    def do_everything(
            self, dim_target: Union[List[int], None], dim_alarm: List[int],
            learning_rate: float, batch_size: int, n_epochs: int, stddev_anomalies: float,
            out_path: Path, evaluation_split: str = "val", plot_freq: int = 0,
            dagmm_conf: dict = None, train_dagmm: bool = True, train_repen: bool = True
    ) -> NoReturn:
        """
        Train & evaluate DA3D and all relevant baseline methods
        :param dim_target: dimensions of the autoencoder's encoder (decoder is symmetric to this)
        :param dim_alarm: dimensions of the alarm network
        :param learning_rate: training learning rate for Adam
        :param batch_size: training batch size
        :param n_epochs: training epochs
        :param stddev_anomalies: standard deviation for anomalies
        :param out_path: output path for the evaluation
        :param plot_freq: plotting frequency
        :param dagmm_conf: special configuration for DAGMM
        :param train_dagmm: train DAGMM
        :param train_repen: train REPEN
        :param evaluation_split: data split to evaluate the methods on
        :return:
        """

        if dagmm_conf is None:
            # This is their setting for KDD - we better stick to their architecture as this 1-dimensional layer seems
            # to make a difference
            dagmm_conf = {
                "comp_hiddens": [12, 4, 1], "comp_activation": tf.nn.tanh,
                "est_hiddens": [10, 2], "est_dropout_ratio": 0.5, "est_activation": tf.nn.tanh,
                "learning_rate": learning_rate, "epoch_size": n_epochs, "minibatch_size": batch_size
            }

        # Train DA3D
        self.train_baseline(
            baseline="AAE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        # If another stddev is given, override the default one
        override_stddev = {}
        if stddev_anomalies is not None:
            override_stddev["sample_stddev"] = stddev_anomalies
        self.train_baseline(
            baseline="DA3D",
            layer_dims=dim_alarm, **override_stddev,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2, "plot_freq": plot_freq}
        )
        self.train_baseline(
            baseline="TrivialSimpleDA3D",
            layer_dims=dim_alarm, **override_stddev,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2}
        )
        # For the ablation study, train DA3D on other stddevs
        for i_da3d in range(1, 5):
            self.train_baseline(
                baseline=f"DA3D-{i_da3d}",
                layer_dims=dim_alarm,
                compile_params={"learning_rate": learning_rate},
                fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2, "plot_freq": plot_freq}
            )

        # Train the baselines
        self.train_baseline(
            baseline="AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="GANomaly",
            enc_dims=dim_target
        )
        if train_dagmm:
            self.train_baseline(
                baseline="DAGMM", **dagmm_conf
            )
        self.train_baseline(
            baseline="A3",
            layer_dims=dim_alarm,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2}
        )
        self.train_baseline(
            baseline="DeepSVDD-AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="DeepSVDD",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="fAnoGAN",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        # REPEN is very slow - stick to the 30 epochs as done in their code
        if train_repen:
            self.train_baseline(
                baseline="REPEN", random_seed=self.random_seed,
                compile_params={"learning_rate": learning_rate},
                fit_params={"epochs": min(n_epochs, 30), "batch_size": batch_size, "verbose": 2},
            )

        # FenceGAN has very specific parameters - better to keep their setup
        if dim_target is not None:
            # Their KDD configuration
            fencegan_latent_dim = 32
            self.train_baseline(
                baseline="FenceGAN",
                input_shape=self.data_setup[0].shape, latent_dim=fencegan_latent_dim,
                compile_params={
                    "G_opt": tf.keras.optimizers.Adam(1e-4, decay=1e-3), "D_opt": tf.keras.optimizers.SGD(8e-6, decay=1e-3)
                },
                fit_params={"batch_size": batch_size, "epochs": n_epochs, "alpha": .5, "gamma": .5}
            )
        else:
            # Their MNIST configuration
            fencegan_latent_dim = 200
            self.train_baseline(
                baseline="FenceGAN",
                latent_dim=fencegan_latent_dim,
                compile_params={
                    "G_opt": tf.keras.optimizers.Adam(2e-5, decay=1e-4), "D_opt": tf.keras.optimizers.Adam(1e-5, decay=1e-4)
                },
                fit_params={"batch_size": batch_size, "epochs": n_epochs, "alpha": .1, "gamma": .1}
            )

        # Get the results
        baseline_methods = {
            "DA3D": {"layer_dims": dim_alarm},
            "TrivialSimpleDA3D": {"layer_dims": dim_alarm},
            "DeepSVDD": {},
            "fAnoGAN": {},
            "GANomaly": {"enc_dims": dim_target},
            "A3": {"layer_dims": dim_alarm},
            "FenceGAN": {"input_shape": self.data_setup[0].shape, "latent_dim": fencegan_latent_dim},
        }
        if train_dagmm:
            baseline_methods["DAGMM"] = dagmm_conf
        if train_repen:
            baseline_methods["REPEN"] = {"random_seed": self.random_seed},
        # Add the ablation study DA3Ds
        for i_da3d in range(1, 5):
            baseline_methods[f"DA3D-{i_da3d}"] = {}
        self.evaluate_ad(
            evaluation_split, out_path=out_path, architecture_params={"layer_dims": dim_alarm},
            evaluate_baseline=baseline_methods
        )

    # -- Helpers --
    @staticmethod
    def parse_name(
            in_conf: dict, prefix: str = None, suffix: str = None,
            is_supervised: bool = False
    ) -> str:
        """
        Convert configuration to a nicer file name
        :param in_conf: dictionary
        :param prefix: a string that will be prepended to the name
        :param suffix: a string that will be appended to the name
        :param is_supervised: is the method supervised? if so use different keywords for the file name
        :return: string describing the dictionary
        """
        # Semi-supervised methods are trained on known anomalies
        keep_keywords = ("y_normal", "y_anomalous", "n_train_anomalies", "p_pollution") if is_supervised \
            else ("y_normal", "p_pollution")

        # Convert to member dict if it's not a dict
        out_dict = in_conf if isinstance(in_conf, dict) else vars(in_conf).copy()

        # Remove all keywords but the desired ones
        out_dict = {
            cur_key: cur_val for cur_key, cur_val in out_dict.items() if cur_key in keep_keywords
        }

        # Parse as string
        out_str = str(out_dict)

        # Remove full stops and others as otherwise the path may be invalid
        out_str = re.sub(r"[{}\\'.<>\[\]()\s]", "", out_str)

        # Alter the string
        if prefix: out_str = prefix + "_" + out_str
        if suffix: out_str = out_str + "_" + suffix

        return out_str

    @staticmethod
    def dict_to_str(in_dict: dict) -> str:
        """
        Parse the values of a dictionary as string
        :param in_dict: dictionary
        :return: dictionary with the same keys but the values as string
        """
        out_dict = {cur_key: str(cur_val) for cur_key, cur_val in in_dict.items()}

        return out_dict

    def get_model_path(
            self, base_path: Path,
            file_name: str = None, file_suffix: str = ".tf",
            sub_folder: str = "", sub_sub_folder: str = "",
    ) -> Path:
        """
        Get the path to save the NN models
        :param base_path: path to the project
        :param file_name: name of the model file (prefix is prepended)
        :param file_suffix: suffix of the file
        :param sub_folder: folder below model folder, e.g. for alarm/target
        :param sub_sub_folder: folder below subfolder, e.g. architecture details
        :return:
        """
        out_path = base_path

        if sub_folder:
            out_path /= sub_folder

        if sub_sub_folder:
            out_path /= sub_sub_folder

        # Create the path if it does not exist
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=False)

        if file_name:
            out_path /= f"{self.save_prefix}_{file_name}"
            out_path = out_path.with_suffix(file_suffix)

        return out_path

