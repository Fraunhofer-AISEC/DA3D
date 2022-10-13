This project is not maintained.
It has been published as part of the following conference paper at IJCNN 2022:
# Double-Adversarial Activation Anomaly Detection
## by Jan-Philipp Schulze, Philip Sperl and Konstantin Böttinger

Anomaly detection is a challenging task for machine learning methods due to the inherent class imbalance.
It is costly and time-demanding to manually analyse the observed data, thus usually only few known anomalies if any are available.
Inspired by generative models and the analysis of the hidden activations of neural networks, we introduce a novel unsupervised anomaly detection method called DA3D.
Here, we use adversarial autoencoders to generate anomalous counterexamples based on the normal data only.
These artificial anomalies used during training allow the detection of real, yet unseen anomalies.
With our novel generative approach, we transform the unsupervised task of anomaly detection to a supervised one, which is more tractable by machine learning and especially deep learning methods.
DA3D surpasses the performance of state-of-the-art anomaly detection methods in a purely data-driven way, where no domain knowledge is required.

### Citation
Double-Adversarial Activation Anomaly Detection was published at IJCNN 2022 [13].
If you find our work useful, please cite the paper:
```
J. -P. Schulze, P. Sperl and K. Böttinger, "Double-Adversarial Activation Anomaly Detection: Adversarial Autoencoders are Anomaly Generators," 2022 International Joint Conference on Neural Networks (IJCNN), 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892896.
```

### Dependencies
We used ``docker`` during the development.
You can recreate our environment by:  
``docker build -t da3d ./docker/``.

Afterwards, start an interactive session while mapping the source folder in the container:  
``docker run --gpus 1 -it --rm -v ~/path/to/da3d/:/app/ -v ~/.keras:/root/.keras da3d``

#### Data sets
The raw data sets are stored in ``./data/``.
You need to add Creditcard [1], Census [2], Cover Type [3], DoH [4], EMNIST [5], NSL-KDD [6], Mammography [7] and URL [8] from the respective website.

For example, the URL's archive contains the file ``All.csv``.
Move it to ``./data/url/All.csv``.
The rest is automatically handled in ``./libs/DataHandler.py``, where you find more information which file is loaded.

#### Baseline methods
The baseline methods are stored in ``./baselines/``.
Whereas we implemented A3, fAnoGAN and Deep-SVDD, you need to add DAGMM [9], GANomaly [10], FenceGAN [11] and REPEN [12] manually from the respective website.

### Instructions

#### Train models
For each data set, all applicable experiments are bundled in the respective ``do_*.py``.
You need to provide a random seed and whether the results should be evaluated on the "val" or "test" data, e.g. ``python ./do_kdd.py 123 val``.
Optional arguments are e.g. the training data pollution ``--p_contamination``.
Please note that we trained the models on a GPU, i.e. there will still be randomness while training the models.
Your models are stored in ``./models/`` if not specified otherwise using ``--model_path``.

#### Evaluate models
After training, the respective models are automatically evaluated on the given data split.
As output, a ``.metric.csv``, ``.roc.csv`` and ``.roc.png`` are given.
By default, these files are stored in ``./models/{p_contamination}_{random_seed}/``.
The first file contains the AUC & AP metrics, the latter two show the ROC curve.
The test results are the merged results of six runs: 
``python3 evaluate_results.py 110 210 310 410 510 610 710``.

### Known Limitations
We sometimes had problems loading the trained models in TensorFlow's eager mode.
Please use graph mode instead.

### File Structure
```
DA3D
│   do_*.py                     (start experiment on the respective data set)
│   evaluate_results.py         (calculate the mean over the test results)
│   README.md                   (file you're looking at)
│
└─── data                       (raw data)
│
└─── docker                     (folder for the Dockerfile)
│
└─── libs
│   └───architecture            (network architecture of the alarm and target networks)
│   └───network                 (helper functions for the NNs)
│   │   DA3D.py                 (main library for our anomaly detection method)
│   │   DataHandler.py          (reads, splits, and manages the respective data set)
│   │   ExperimentWrapper.py    (wrapper class to generate comparable experiments)
│   │   Metrics.py              (methods to evaluate the data)
│
└─── models                     (output folder for the trained neural networks)
│
└─── baselines                  (baseline methods)
│
```

### Links
* [1] https://www.kaggle.com/mlg-ulb/creditcardfraud
* [2] https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
* [3] https://archive.ics.uci.edu/ml/datasets/Covertype
* [4] https://www.unb.ca/cic/datasets/dohbrw-2020.html
* [5] https://www.nist.gov/itl/products-and-services/emnist-dataset
* [6] https://www.unb.ca/cic/datasets/nsl.html
* [7] http://odds.cs.stonybrook.edu/mammography-dataset/
* [8] https://www.unb.ca/cic/datasets/url-2016.html
* [9] https://github.com/tnakae/DAGMM
* [10] https://github.com/chychen/tf2-ganomaly
* [11] https://github.com/phuccuongngo99/Fence_GAN
* [12] https://github.com/GuansongPang/deep-outlier-detection
* [13] https://ieeexplore.ieee.org/abstract/document/9892896

