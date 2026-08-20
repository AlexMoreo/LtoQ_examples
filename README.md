# Learning to Quantify: some examples

This repo contains some code examples showcasing basic quantification methods in action,
including classic ones like CC, ACC, SLD, HDy, and HDx, and modern ones like
KMM, KDEy, or HistNet.

The base code is based on the [QuaPy](https://github.com/HLT-ISTI/QuaPy) framework.

There are three main scripts:
* plotting_diagonal_CCvariants.py: showcases the bias of the unadjusted classify-and-count towards
  the training prevalence
* plotting_diagonal_and_errdrift.py: showcases classical methods' errors as a function of the amount of shift
* over_time_experiment.py: showcases different quantifiers in a sentiment quantification task
    where prevalence values evolve over time, based on a Kaggle dataset


## Run the experiments

First, let us configure the environment 

```bash
conda create -n quant python=3.11
conda activate quant
pip install quapy[neural]
pip install transformers
pip install kugglehub
pip install --upgrade pip setuptools wheel
pip install "jax[cpu]"
pip install "qunfold @ git+https://github.com/mirkobunse/qunfold@v0.1.5"
```

Then run the scripts without args, e.g.:

```bash
python over_time_experiment.py
```

