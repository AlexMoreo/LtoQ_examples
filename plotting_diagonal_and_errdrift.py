from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV

import quapy as qp
from quapy.data import LabelledCollection
from method.non_aggregative import DMx
from protocol import APP
from quapy.method.aggregative import CC, DMy, ACC, EMQ
from tqdm import tqdm

qp.environ['SAMPLE_SIZE'] = 500
training_prevalence = 0.1


def cls():
    return LogisticRegressionCV(n_jobs=-1,Cs=10)

def gen_methods():
    yield CC(cls()), r'CC$_{10' + r'\%}$'
    yield ACC(cls()), 'ACC'
    yield EMQ(cls()), 'SLD'
    yield DMy(cls(), val_split=10, nbins=10, n_jobs=-1), 'HDy'
    yield DMx(nbins=10, n_jobs=-1), 'HDx'

def gen_data():

    train, test = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=5).train_test

    method_data = []
    training_size = 5000
    # since the problem is binary, it suffices to specify the negative prevalence, since the positive is constrained
    train_sample = train.sampling(training_size, 1-training_prevalence, random_state=0)

    for model, method_name in tqdm(gen_methods(), total=4):
        with qp.util.temp_seed(1):
            if method_name == 'HDx':
                svd = TruncatedSVD(n_components=5, random_state=0)
                Xtr_red = svd.fit_transform(train_sample.X)
                model.fit(Xtr_red, train_sample.y)

                test_dense = LabelledCollection(svd.transform(test.X), test.y)
                true_prev, estim_prev = qp.evaluation.prediction(model, APP(test_dense, repeats=100, random_state=0))
            else:
                model.fit(*train_sample.Xy)
                true_prev, estim_prev = qp.evaluation.prediction(model, APP(test, repeats=100, random_state=0))
        method_data.append((method_name, true_prev, estim_prev, train_sample.prevalence()))

    return zip(*method_data)


method_names, true_prevs, estim_prevs, tr_prevs = gen_data()

qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, train_prev=tr_prevs[0], savepath='./plots/diagonalplot_classic_methods.pdf')
qp.plot.error_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, n_bins=10, savepath='./plots/errdriftplot_classic_methods.pdf', title='', show_density=False, show_std=True)
