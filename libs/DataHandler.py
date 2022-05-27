import abc
import math

import numpy as np
import pandas as pd
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff, loadmat

from typing import List, Callable, Union, Tuple, Dict, NoReturn

from libs.DataTypes import ExperimentData
from libs.constants import BASE_PATH


class DataLabels:
    """
    Class storing test/train data
    """

    def __init__(
            self, x_train: np.ndarray, y_train: np.ndarray,
            y_normal: Union[List[int], List[str]], y_anomalous: Union[List[int], List[str]],
            x_test: np.ndarray = None, y_test: np.ndarray = None,
            x_val: np.ndarray = None, y_val: np.ndarray = None,
            p_test: float = .15, p_val: float = .05, p_pollution: float = 0.0, n_train_anomalies: int = None,
            allow_unused: bool = False,
            random_state: int = None
    ):

        # We'll put everything in the train data if no test data was given and split later
        self.x_train: np.ndarray = x_train  # Train data
        self.y_train: np.ndarray = y_train
        self.x_test: np.ndarray = x_test  # Test data
        self.y_test: np.ndarray = y_test
        self.x_val: np.ndarray = x_val  # Validation data
        self.y_val: np.ndarray = y_val

        # Store normal and anomalous data globally
        self.y_normal: Union[List[int], List[str]] = y_normal
        self.y_anomalous: Union[List[int], List[str]] = y_anomalous
        self.y_all = list(set(y_normal) | set(y_anomalous))
        # Sanity check: we should not have overlapping classes
        assert len(self.y_all) == len(y_normal + y_anomalous), "Normal and anomalous classes must not overlap"

        # If needed: a scaler
        self.scaler: TransformerMixin = None

        # Configuration
        self.test_split: float = p_test  # Test data percentage
        self.val_split: float = p_val  # Validation data percentage
        self.p_pollution: float = p_pollution
        self.n_train_anomalies: int = n_train_anomalies
        self.random_state = random_state
        self.random_gen = np.random.default_rng(random_state)
        self.allow_unused = allow_unused

        # Metadata
        self.shape: tuple = None  # Shape of the data

        # Fill the values
        self._post_init()

    ## Class methods
    def __repr__(self):
        return self.__class__.__name__

    ## Retrievers
    def get_target_autoencoder_data(
            self, data_split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for autoencoders
        :param data_split: get data of either "train", "val" or "test"
        :param drop_classes: which classes to drop, drop none if None
        :param include_classes: which classes to include (has priority over drop_classes)
        :param p_contaminate: fraction of contaminated samples, i.e. dropped samples added to the output
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)
        this_x = this_data[0]

        # For the autoencoder, we don't need much else than x
        return this_x, this_x

    def get_target_classifier_data(
            self, data_split: str, only_normal: bool = True, cast_str: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for classifiers
        :param data_split: get data of either "train", "val" or "test"
        :param only_normal: get labels of normal data points only
        :param cast_str: cast str labels to numbers
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split, only_normal=only_normal)
        this_x = this_data[0]
        this_y = this_data[1]

        # Return the data
        return this_x, this_y

    def get_alarm_data(
            self, data_split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the labels for the alarm network, i.e. with binary anomaly labels
        :param data_split: get data of either "train", "val" or "test"
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split, only_normal=False)
        this_x = this_data[0]

        # Make labels binary
        this_y = np.isin(this_data[1], self.y_anomalous)
        this_y = this_y.astype("uint8")

        return this_x, this_y

    @staticmethod
    def _idx_anom_samples(y: np.ndarray, n_anomaly_samples: int = None) -> np.ndarray:
        """
        Give the indices that should be deleted due to less anomaly samples
        :param y: binary array
        :param n_anomaly_samples: amount of anomaly samples that should be kept
        :return: indices about to be deleted
        """
        # Don't delete anything if None is given
        if n_anomaly_samples is None:
            return np.array([])

        # IDs of all anomaly samples
        idx_anom = np.where(y == 1)[0]

        # Select the indices to delete
        n_delete = len(idx_anom) - n_anomaly_samples
        # assert n_delete > 0  # Not true in the unsupervised case where we don't use any
        idx_delete = np.random.choice(idx_anom, size=n_delete, replace=False)

        return idx_delete

    def get_attention_labels(
            self, data_split: str, add_global: bool = True
    ) -> np.ndarray:
        """
        Get the labels for the attention network, i.e. the position of the target networks
        :param data_split: get data of either "train", "val" or "test"
        :param add_global: add an extra column for the anomaly expert
        :return: labels
        """

        # Note down all available labels
        all_labels = self.y_normal.copy()
        anomaly_label = "anom" if isinstance(all_labels[0], str) else -1
        assert anomaly_label not in all_labels, "The used anomaly label may overwrite existing ones"
        if add_global:
            all_labels.append(anomaly_label)

        # We transform the classification labels to a 1-Hot matrix
        y_cat = self.get_target_classifier_data(data_split=data_split, only_normal=False)[1]
        # Important: must be signed int, otherwise -1 = 255 or whatever the range is
        if anomaly_label == -1:
            y_cat = y_cat.astype(np.int16)
        # Summarise all known anomalies to one "other" class
        y_cat[np.where(np.isin(y_cat, self.y_anomalous))] = anomaly_label
        # Use Pandas for easier mapping
        y_cat = pd.Series(y_cat[:, 0], dtype=pd.api.types.CategoricalDtype(categories=all_labels, ordered=False))
        this_y = pd.get_dummies(y_cat)

        return this_y.to_numpy()

    def get_mae_data(
            self, data_split: str, equal_size: bool = True
    ) -> Dict[Union[int, str], tuple]:
        """
        Get multi-autoencoder data, i.e. AE data distributed among experts
        :param data_split: get data of either "train", "val" or "test"
        :param equal_size: equalise the amount of samples for each expert by oversampling, i.e. we repeat samples in the smaller sets
        :return: features and labels for each data class
        """

        # We basically loop through all normal labels
        this_x, this_y = self.get_target_classifier_data(data_split=data_split, only_normal=True)

        expert_data = {}
        for cur_label in self.y_normal:
            idx_label = np.where(this_y == cur_label)[0]
            expert_data[cur_label] = (this_x[idx_label, :], this_x[idx_label, :])

        # If we want to equalise the amount of samples, we'll repeat some of them
        if equal_size:
            expert_data = self.equalise_expert_data(expert_data)

        return expert_data

    ## Preprocessors
    def _post_init(self):
        """
        Process the data
        :return:
        """

        # Check if all classes are covered
        available_classes = np.unique(self.y_train).tolist()
        assert (self.allow_unused or (set(available_classes) == set(self.y_all))), \
            "There are classes in training data are not covered by the normal&anomalous samples. " \
            f"Is this intended? All classes: {available_classes}."

        # The labels should not have an empty dimension
        if len(self.y_train.shape) == 1:
            self.y_train = np.expand_dims(self.y_train, axis=-1)
        if (self.y_val is not None) and (len(self.y_val.shape) == 1):
            self.y_val = np.expand_dims(self.y_val, axis=-1)
        if (self.y_test is not None) and (len(self.y_test.shape) == 1):
            self.y_test = np.expand_dims(self.y_test, axis=-1)

        # Split in test and train
        # Fix the test to one seed to simulate the "split and forget" nature of it
        if self.x_test is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, test_size=self.test_split, random_state=42
            )

        # Split in train and validation
        if self.x_val is None:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train, self.y_train, test_size=self.val_split, random_state=self.random_state
            )

        # Print statistics
        self._print_statistics()
        # Pollute the data
        if self.p_pollution: self._pollute_data()
        # Reduce training anomalies
        if self.n_train_anomalies is not None: self._reduce_train_anomalies()
        # Preprocess
        self._preprocess()
        # Note down the shape
        self.shape = self.x_train.shape[1:]

    @abc.abstractmethod
    def _preprocess(self):
        # Preprocessing steps, e.g. data normalisation
        raise NotImplementedError("Implement in subclass")

    def _print_statistics(self) -> NoReturn:
        """
        Show statistics, e.g. the class count.
        :return:
        """
        n_train_norm = np.sum(np.isin(self.y_train, self.y_normal))
        n_test_norm = np.sum(np.isin(self.y_test, self.y_normal))
        n_test_anom = np.sum(np.isin(self.y_test, self.y_anomalous))

        print("Class counts:")
        print(f"Latex: & \\num{{{n_train_norm:.2e}}} & \\num{{{n_test_norm:.2e}}} & \\num{{{n_test_anom:.2e}}}")
        print(f"Train normal: {n_train_norm} \t Test normal: {n_test_norm} \t Test Anomalous: {n_test_anom}")

    def _pollute_data(self) -> NoReturn:
        """
        Change the label of some anomalous training samples to random normal training labels
        :return:
        """
        # Filter for normal an anomalous samples
        idx_normal = np.where(np.isin(self.y_train, self.y_normal))[0]
        idx_anomalous = np.where(np.isin(self.y_train, self.y_anomalous))[0]

        # Calculate how many samples we need for the pollution
        n_pollute = self._contaminate_p_to_n(self.p_pollution, idx_normal.shape[0])
        # Check if we have enough samples
        need_duplicates = n_pollute > idx_anomalous.shape[0]
        if need_duplicates:
            print(f"For the desired pollution level, {n_pollute} samples are required, "
                  f"but there are only {len(idx_anomalous)} known anomalies. There will be duplicates.")

        # Assign random normal labels to some of the anomalous samples
        idx_pollute = self.random_gen.choice(idx_anomalous, n_pollute, replace=need_duplicates)
        self.y_train[idx_pollute, :] = self.random_gen.choice(self.y_normal, self.y_train[idx_pollute].shape, replace=True)

    def _reduce_train_anomalies(self) -> NoReturn:
        """
        Use less anomaly samples during training
        :return:
        """
        # Filter for normal an anomalous samples
        idx_normal = np.where(np.isin(self.y_train, self.y_normal))[0]
        idx_anomalous = np.where(np.isin(self.y_train, self.y_anomalous))[0]

        # Reduce the number of anomalies
        idx_anomalous = self.random_gen.choice(idx_anomalous, self.n_train_anomalies, replace=False)
        # Concatenate with normal samples to get all samples to keep
        idx_keep = np.concatenate([idx_normal, idx_anomalous], axis=0)

        # Filter
        self.x_train = self.x_train[idx_keep, :]
        self.y_train = self.y_train[idx_keep, :]

        pass

    ## Helpers
    def include_to_drop(self, include_data: Union[List[int], List[str]]) -> Union[List[int], List[str]]:
        """
        Convert a list of classes to include to a list of classes to drop
        :param include_data: classes to include
        :param all_classes: available classes
        :return: classes to drop
        """

        drop_classes = set(self.available_classes) - set(include_data)

        return list(drop_classes)

    @staticmethod
    def equalise_expert_data(expert_data: dict) -> dict:
        """
        Equalise the length of all expert clusters by repeating the entries
        :param expert_data: dictionary indexed by the expert clusters
        :return: dictionary indexed by the expert clusters, but equally sized data
        """
        max_size = max([cur_val[0].shape[0] for cur_val in expert_data.values()])

        for cur_idx, cur_val in expert_data.items():
            # Nothing to equalise if it's the same size
            if cur_val[0].shape[0] == max_size:
                continue

            # Sample random items and add the to the data
            cur_size = cur_val[0].shape[0]
            idx_repeat = np.random.choice(cur_size, size=max_size - cur_size)

            extended_x = np.concatenate(
                [cur_val[0], cur_val[0][idx_repeat, :]], axis=0
            )
            extended_y = np.concatenate(
                [cur_val[1], cur_val[1][idx_repeat, :]], axis=0
            )

            expert_data[cur_idx] = (extended_x, extended_y)

        return expert_data

    def to_data(
            self, train_type: str = "train", test_type: str = "test",
            for_experts: bool = False, equal_size: bool = False
    ) -> ExperimentData:
        """
        Convert the configuration to actual data
        :param train_type: use the train or validation data for training (only used to load less data while debugging)
        :param test_type: use the test or validation data for evaluation (i.e. code once, use twice)
        :param for_experts: split the target data (one class for each expert)
        :param equal_size: equalise the amount of samples for each expert by oversampling, i.e. we repeat samples in the smaller sets
        """

        return ExperimentData(
            # Target training: all normal samples
            train_target=self.get_mae_data(
                data_split=train_type, equal_size=equal_size,
            ) if for_experts else self.get_target_autoencoder_data(
                data_split=train_type
            ),
            train_alarm=self.get_alarm_data(
                data_split=train_type
            ),
            val_target=self.get_mae_data(
                data_split="val", equal_size=equal_size,
            ) if for_experts else self.get_target_autoencoder_data(
                data_split="val"
            ),
            val_alarm=self.get_alarm_data(
                data_split="val"
            ),
            # Target testing: all normal samples plus the test anomalies
            test_target=self.get_mae_data(
                data_split=test_type, equal_size=equal_size,
            ) if for_experts else self.get_target_autoencoder_data(
                data_split=test_type
            ),
            test_alarm=self.get_alarm_data(
                data_split=test_type
            ),
            # Shape to generate networks
            data_shape=self.shape,
            input_shape=(None, ) + self.shape
        )

    @staticmethod
    def _contaminate_p_to_n(p_contaminate: float, n_samples: int) -> int:
        """
        Determine the amount of samples needed to be added such that they make up p_train_contamination% of the resulting set
        :param p_contaminate: fraction of contaminated samples in the resulting data set
        :param n_samples: number of samples so far
        :return: number of contaminated samples that need to be added
        """

        assert 0 < p_contaminate < 1

        # Oh dear, percentages: we need to add a higher fraction of samples to get the desired fraction in the new set
        p_add = p_contaminate / (1-p_contaminate)
        n_contaminate = round(n_samples * p_add)

        return n_contaminate

    def _get_data_set(self, data_split: str, only_normal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the right data split
        :param data_split: train, val or test data?
        :param only_normal: return normal data only
        :return: the right data set
        """

        if data_split == "train":
            this_data = (self.x_train.copy(), self.y_train.copy())

        elif data_split == "test":
            this_data = (self.x_test.copy(), self.y_test.copy())

        elif data_split == "val":
            this_data = (self.x_val.copy(), self.y_val.copy())

        else:
            raise ValueError("The requested data must be of either train, val or test set.")

        # Maybe filter for normal data
        if only_normal:
            idx_normal = np.where(np.isin(this_data[1], self.y_normal))[0]
            this_data = (this_data[0][idx_normal, :], this_data[1][idx_normal, :])

        return this_data

    def scikit_scale(self, scikit_scaler: Callable[[], TransformerMixin] = MinMaxScaler):
        """
        Apply a scikit scaler to the data, e.g. MinMaxScaler transform data to [0,1]
        :return:
        """
        # Fit scaler to train set
        self.scaler = scikit_scaler()
        self.x_train = self.scaler.fit_transform(self.x_train)

        # Scale the rest
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        pass


class MNIST(DataLabels):
    def __init__(
            self, special_name: str = None,
            path_emnist: Path = (BASE_PATH / "data" / "EMNIST" / "emnist-letters").with_suffix(".mat"),
            *args, **kwargs
    ):
        """
        Load the (fashion) MNIST data set
        EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
        :param special_name: load an MNIST-like data set instead (e.g. emnist or fashion), use good old MNIST if None
        """
        assert special_name in ["emnist", "fashion"] or special_name is None

        # Simply load the data with the kind help of Keras
        if special_name == "fashion":
            this_data = tf.keras.datasets.fashion_mnist.load_data()
        elif special_name == "emnist":
            # Load from file, inspired by extra-keras-datasets
            this_data = loadmat(path_emnist)
            # The data is hidden in there
            this_data = this_data["dataset"]
            this_data = (
                (
                    np.reshape(this_data["train"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"),
                    this_data["train"][0, 0]["labels"][0, 0]
                ),
                (
                    np.reshape(this_data["test"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"),
                    this_data["test"][0, 0]["labels"][0, 0]
                )
            )
        else:
            this_data = tf.keras.datasets.mnist.load_data()
        # Extract the data parts
        (x_train, y_train), (x_test, y_test) = this_data

        # Add channel dimension to the data
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        super(MNIST, self).__init__(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, *args, **kwargs
        )

    def _preprocess(self):
        """
        For MNIST, we can scale everything by just dividing by 255
        :return:
        """
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.
        self.x_val = self.x_val / 255.


class CreditCard(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "creditcard" / "creditcard").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the CreditCard data set (https://www.kaggle.com/mlg-ulb/creditcardfraud)
        :param data_path: absolute path to the CreditCard csv
        """

        data = pd.read_csv(data_path)

        # Time axis does not directly add information (although frequency might be a feature)
        data = data.drop(['Time'], axis=1)

        # Column class has the anomaly values, the rest is data
        y_train = data.pop("Class")

        # Bring y in the right shape
        y_train = np.reshape(y_train.to_numpy(), (-1, ))

        super(CreditCard, self).__init__(
            x_train=data.to_numpy(), y_train=y_train, *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """

        self.scikit_scale()


class CovType(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "covtype" / "covtype").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the covertype data set (https://archive.ics.uci.edu/ml/datasets/Covertype)
        :param data_path: absolute path to the covtype csv
        """

        # Open raw data
        train_data = pd.read_csv(data_path)

        x_train = train_data.drop(columns="Cover_Type")
        y_train = train_data.loc[:, "Cover_Type"]

        super(CovType, self).__init__(
            x_train=x_train.to_numpy(), y_train=y_train.to_numpy(), *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """

        self.scikit_scale()


class Mammography(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "mammography" / "mammography").with_suffix(".mat"),
            *args, **kwargs
    ):
        """
        Load the mammography data set (http://odds.cs.stonybrook.edu/mammography-dataset/)
        :param data_path: absolute path to the mammography.mat
        """

        # Open raw data
        train_data = loadmat(data_path)
        # Extract data
        x = train_data["X"]
        # We take our labels to be scalar
        y = train_data["y"]
        y = np.reshape(y, (-1, ))

        super(Mammography, self).__init__(
            x_train=x, y_train=y, *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """

        self.scikit_scale()


class URL(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "url" / "All").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the URL 2016 data set (https://www.unb.ca/cic/datasets/url-2016.html)
        :param data_path: absolute path to the All.csv
        """

        # Open raw data
        train_data = pd.read_csv(data_path)

        # There are some NaN entries: we'll drop the columns to be more data-efficient
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(axis=1)

        # The labels are given by the last column
        y_train = train_data.pop("URL_Type_obf_Type")

        # We further divide the normal class among the TLDs
        # tld, domain_token_count, this.fileExtLen, longdomaintokenlen, ldl_domain, dld_url,
        y_train[y_train == "benign"] = train_data.loc[y_train == "benign", "ldl_domain"].astype("str")

        super(URL, self).__init__(
            x_train=train_data.to_numpy(), y_train=y_train.to_numpy(),
            *args, **kwargs
        )

    def _preprocess(self):

        self.scikit_scale()


class DoH(DataLabels):
    def __init__(
            self, data_path: Path = BASE_PATH / "data" / "doh",
            *args, **kwargs
    ):
        """
        Load the DoH data set (https://www.unb.ca/cic/datasets/dohbrw-2020.html)
        :param data_path: path to the raw .csv files
        """

        # Open raw data
        train_data = [
            pd.read_csv((data_path / "l2-benign").with_suffix(".csv"), index_col="TimeStamp", parse_dates=True),
            pd.read_csv((data_path / "l2-malicious").with_suffix(".csv"), index_col="TimeStamp", parse_dates=True),
        ]
        train_data = pd.concat(train_data)

        # Drop non-informative labels like the IP
        train_data = train_data.drop(columns=[
            "SourceIP", "DestinationIP", "SourcePort", "DestinationPort"
        ])

        # There are some NaN entries: we'll drop the rows to be more data-efficient
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(axis=0)

        # The labels are given by the last column
        y_train = train_data.pop("Label")
        # Divide the normal class by the week number
        train_data["week"] = train_data.index.week
        y_train[y_train == "Benign"] = train_data.loc[y_train == "Benign", "week"].astype("str")
        # Note: we leave the week in the data so that the other methods also have a chance to learn the correlation
        # y_week = train_data.pop("week")

        super(DoH, self).__init__(
            x_train=train_data.to_numpy(), y_train=y_train.to_numpy(),
            *args, **kwargs
        )

    def _preprocess(self):

        self.scikit_scale()


class Census(DataLabels):
    def __init__(
            self, data_path: Path = BASE_PATH / "data" / "census",
            *args, **kwargs
    ):
        """
        Load the census data set (https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD))
        :param data_path: path where the training and test data resides
        """

        # Open raw training data and convert to categorical what is categeorical
        train_data = pd.read_csv(
            data_path / "census-income.test", header=None,
            dtype={
                # Column 24 does not seem to be in the readme
                cur_idx: "category" for cur_idx in list(set(range(42)) - set([0, 5, 16, 17, 18, 24, 30, 39]))
            }
        )
        # Copy the dtypes to the test data to get the same 1-Hot encoding later
        test_data = pd.read_csv(data_path / "census-income.test", header=None, dtype=train_data.dtypes.to_dict())

        # We'll take the income as anomaly label and the sex as normal classes
        y_train = train_data.iloc[:, 12].astype("str").str.strip()
        y_test = test_data.iloc[:, 12].astype("str").str.strip()
        y_train_income = train_data.pop(41)
        y_test_income = test_data.pop(41)
        # Let "50000+" be our anomaly
        y_train[y_train_income == " 50000+."] = "Anomalous"
        y_test[y_test_income == " 50000+."] = "Anomalous"

        # 1-Hot encode
        train_data = pd.get_dummies(train_data)
        test_data = pd.get_dummies(test_data)
        assert (train_data.columns == test_data.columns).all()

        super(Census, self).__init__(
            x_train=train_data.to_numpy(), y_train=y_train.to_numpy(),
            x_test=test_data.to_numpy(), y_test=y_test.to_numpy(),
            *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """

        self.scikit_scale()


class NSL_KDD(DataLabels):
    def __init__(self, data_folder: str = "NSL-KDD", *args, **kwargs):
        """
        NSL KDD data set: https://www.unb.ca/cic/datasets/nsl.html
        :param data_folder: subfolder of "data" where raw data resides
        """

        # Open raw data
        common_path = BASE_PATH / "data" / data_folder
        train_data = arff.loadarff((common_path / "KDDTrain+").with_suffix(".arff"))
        test_data = arff.loadarff((common_path / "KDDTest+").with_suffix(".arff"))

        # Extract column names
        all_cols = [cur_key for cur_key in test_data[1]._attributes.keys()]
        all_cat = {
            cur_key: cur_val.range for cur_key, cur_val in test_data[1]._attributes.items()
            if cur_val.range is not None
        }

        # Create pandas dataframe
        train_data = pd.DataFrame(data=train_data[0], columns=all_cols)
        test_data = pd.DataFrame(data=test_data[0], columns=all_cols)

        # Mark respective columns as categorical
        for cur_key, cur_val in all_cat.items():
            # We need to decode the byte strings first
            test_data[cur_key] = pd.Categorical(
                test_data[cur_key].str.decode('UTF-8'), categories=cur_val, ordered=False
            )
            train_data[cur_key] = pd.Categorical(
                train_data[cur_key].str.decode('UTF-8'), categories=cur_val, ordered=False
            )

        # Drop the class labels from the original data
        train_labels = train_data.pop("class").astype("str")
        test_labels = test_data.pop("class").astype("str")

        # Use "protocol_type" to split the normal class
        # Other choices: protocol_type, land, service, logged_in, root_shell, is_host_login, is_guest_login
        train_labels.loc[train_labels == "normal"] = train_data.loc[train_labels == "normal", "protocol_type"]
        test_labels.loc[test_labels == "normal"] = test_data.loc[test_labels == "normal", "protocol_type"]

        # Finally, 1-Hot encode the categorical data
        train_data = pd.get_dummies(train_data)
        test_data = pd.get_dummies(test_data)
        assert (train_data.columns == test_data.columns).all()

        # We'll use ndarrays from now on
        super(NSL_KDD, self).__init__(
            x_train=train_data.to_numpy(), y_train=train_labels.to_numpy(),
            x_test=test_data.to_numpy(), y_test=test_labels.to_numpy(), *args, **kwargs
        )

    def _attack_map(self) -> dict:
        """
        Map grouping the single attack classes
        :return: mapping dictionary
        """

        attack_dict = {
            'normal': 'normal',

            'back': 'DoS',
            'land': 'DoS',
            'neptune': 'DoS',
            'pod': 'DoS',
            'smurf': 'DoS',
            'teardrop': 'DoS',
            'mailbomb': 'DoS',
            'apache2': 'DoS',
            'processtable': 'DoS',
            'udpstorm': 'DoS',

            'ipsweep': 'Probe',
            'nmap': 'Probe',
            'portsweep': 'Probe',
            'satan': 'Probe',
            'mscan': 'Probe',
            'saint': 'Probe',

            'ftp_write': 'R2L',
            'guess_passwd': 'R2L',
            'imap': 'R2L',
            'multihop': 'R2L',
            'phf': 'R2L',
            'spy': 'R2L',
            'warezclient': 'R2L',
            'warezmaster': 'R2L',
            'sendmail': 'R2L',
            'named': 'R2L',
            'snmpgetattack': 'R2L',
            'snmpguess': 'R2L',
            'xlock': 'R2L',
            'xsnoop': 'R2L',
            'worm': 'R2L',

            'buffer_overflow': 'U2R',
            'loadmodule': 'U2R',
            'perl': 'U2R',
            'rootkit': 'U2R',
            'httptunnel': 'U2R',
            'ps': 'U2R',
            'sqlattack': 'U2R',
            'xterm': 'U2R'
        }

        return attack_dict

    def _preprocess(self):
        """
        Minmaxscale the data
        :return:
        """
        self.scikit_scale(scikit_scaler=MinMaxScaler)

