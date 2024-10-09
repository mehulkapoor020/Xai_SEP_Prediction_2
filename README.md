## Predicting Solar Energetic Particles Using SDO/HMI Vector Magnetic Data Products and a Bidirectional LSTM Network<br>
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/sep.svg)](https://zenodo.org/record/7516609#.Y7xChBXMLrk)


## Authors
Yasser Abduallah, Vania K. Jordanova, Hao Liu, Qin Li, Jason T. L. Wang, and Haimin Wang

## Abstract

Solar energetic particles (SEPs) are an essential source of space radiation, and are hazardous for humans in space,
spacecraft, and technology in general. In this paper, we propose a deep-learning method, specifically a bidirectional
long short-term memory (biLSTM) network, to predict if an active region (AR) would produce an SEP event given
that (i) the AR will produce an M- or X-class flare and a coronal mass ejection (CME) associated with the flare, or
(ii) the AR will produce an M- or X-class flare regardless of whether or not the flare is associated with a CME. The
data samples used in this study are collected from the Geostationary Operational Environmental Satelliteʼs X-ray
flare catalogs provided by the National Centers for Environmental Information. We select M- and X-class flares
with identified ARs in the catalogs for the period between 2010 and 2021, and find the associations of flares,
CMEs, and SEPs in the Space Weather Database of Notifications, Knowledge, Information during the same period.
Each data sample contains physical parameters collected from the Helioseismic and Magnetic Imager on board the
Solar Dynamics Observatory. Experimental results based on different performance metrics demonstrate that the
proposed biLSTM network is better than related machine-learning algorithms for the two SEP prediction tasks
studied here. We also discuss extensions of our approach for probabilistic forecasting and calibration with
empirical evaluation.


For the latest updates of the tool refer to https://github.com/deepsuncode/SEP-prediction

## Installation on local machine
To install TensorFlow with pip refer to https://www.tensorflow.org/install/pip

Tested on Python 3.9.16 and the following version of libraries
|Library | Version   | Description  |
|---|---|---|
| tensorflow| 2.10.1| Deep learning tool for high performance computation |
| tensorboard| 2.10.1| Provides the visualization and tooling needed for machine learning|
|keras| 2.10.0 | Deep learning API|
|scikit-learn| 1.2.1| Machine learning|
| pandas|1.5.3| Data loading and manipulation|
|matplotlib|3.6.3|Plotting and graphs|
|numpy| 1.24.2| Array manipulation|
