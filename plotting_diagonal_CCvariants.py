from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

import quapy as qp
from protocol import APP
from quapy.method.aggregative import CC
from sklearn.svm import LinearSVC
import numpy as np
from tqdm import tqdm

qp.environ['SAMPLE_SIZE'] = 500

def gen_data():

    train, test = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=5).train_test

    method_data = []
    for training_prevalence in tqdm(np.linspace(0.1, 0.9, 9), total=9):
        training_size = 5000
        # since the problem is binary, it suffices to specify the negative prevalence, since the positive is constrained
        train_sample = train.sampling(training_size, 1-training_prevalence)

        # cls = GridSearchCV(LinearSVC(), param_grid={'C': np.logspace(-2,2,5), 'class_weight':[None, 'balanced']}, n_jobs=-1)
        # cls = GridSearchCV(LogisticRegression(), param_grid={'C': np.logspace(-2, 2, 5), 'class_weight': [None, 'balanced']}, n_jobs=-1)
        # cls.fit(*train_sample.Xy)

        model = CC(LogisticRegressionCV(n_jobs=-1,Cs=10))

        model.fit(*train_sample.Xy)
        true_prev, estim_prev = qp.evaluation.prediction(model, APP(test, repeats=100, random_state=0))
        method_name = 'CC$_{'+f'{int(100*training_prevalence)}' + '\%}$'
        method_data.append((method_name, true_prev, estim_prev, train_sample.prevalence()))

    return zip(*method_data)


method_names, true_prevs, estim_prevs, tr_prevs = gen_data()

qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, savepath='./plots/bin_diag_cc.pdf')
# qp.plot.error_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, n_bins=10, savepath='./plots_cacm/err_drift_cc.pdf', title='', show_density=False)
