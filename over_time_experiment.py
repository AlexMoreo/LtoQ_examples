import os
from collections import defaultdict

import kagglehub
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from qunfold import KMM
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from quapy.method.meta import HistNetQ
from quapy.data import LabelledCollection
import quapy.functional as F
from method.composable import QUnfoldWrapper
from quapy.method.aggregative import CC, ACC, DistributionMatchingY, EMQ, KDEyML
from quapy.method.non_aggregative import DistributionMatchingX
from transformers import AutoTokenizer, AutoModel
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.precision", 4)
pd.set_option("display.float_format", "{:.4f}".format)

# ------------------------------------------------------------------------------------
# Data loader and preprocessing
# ------------------------------------------------------------------------------------
def prepare_xy_date_blocks(df, freq="M"):
    """
    df: DataFrame with columns 'text', 'airline_sentiment', 'tweet_created'
    freq: frequency of temporal blocks, i.e., day ('D'), week ('W'), month ('M'), etc.

    Returns:
        X: list of texts
        y: np.ndarray of labels
        date: list of int indexes per temporal batch
        idx2date: list with temporal bounds (tuple) for each batch
    """

    df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce")
    df = df.sort_values("tweet_created").reset_index(drop=True)

    X = df["text"].astype(str).values
    y = df["airline_sentiment"].values

    # group dates by requested frequency
    date_groups = df["tweet_created"].dt.to_period(freq)

    # assigns index to date blocks
    unique_periods = date_groups.unique()
    period_to_idx = {p: i for i, p in enumerate(unique_periods)}

    date = np.asarray([period_to_idx[p] for p in date_groups])

    # get true limits of period intervals
    idx2date = []
    for p in unique_periods:
        start = p.start_time
        end = p.end_time
        idx2date.append((start, end))

    return X, y, date, idx2date


def prepare_labelled_collections(filter_neutral):
    # loads and prepares the Twitter US Airlines Sentiment dataset (from Kaggle)
    # returns a labelled collection for the training data (day 0 and 1), and a list of the test
    # sets (day 2 onwards) and the time limits for each test period
    # The dataset is originally ternary (negative, neutral, positive), but we binarize it discarding neutral

    # Download latest version
    path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
    df = pd.read_csv(Path(path) / 'Tweets.csv')
    X, y, date, idx2date = prepare_xy_date_blocks(df, freq="D")

    # binarize
    if filter_neutral:
        keep_idx = (y!='neutral')
        X = X[keep_idx]
        y = y[keep_idx]
        date = date[keep_idx]
    else:
        y[y == 'neutral'] = 2
    y[y == 'positive'] = 1
    y[y == 'negative'] = 0
    y = y.astype(int)

    # use day 0 for training, the rest for test
    X_train, y_train = X[date<=1], y[date<=1]
    train = LabelledCollection(X_train, y_train)
    print(f'training has {len(train)} docs and prevalence={F.strprev(train.prevalence())} classes={train.classes}')

    tests = []
    test_init = []
    for date_i in range(2, max(date)+1):
        X_test_i, y_test_i = X[date==date_i], y[date==date_i]
        test_i = LabelledCollection(X_test_i, y_test_i, classes=train.classes)
        print(f'test-{date_i} has {len(test_i)} docs and prevalence={F.strprev(test_i.prevalence())}')
        tests.append(test_i)
        test_init.append(idx2date[date_i])

    return train, tests, test_init


# ------------------------------------------------------------------------------------
# HuggingFace wrapper for document embedding
# ------------------------------------------------------------------------------------
class HFEmbedder:
    """
    Extracts a fixed-size sentence embedding per document from a frozen (not fine-tuned) huggingface
    transformer encoder, via mean-pooling of the last hidden state over the non-padding tokens.
    Used by embedded_representation() below.
    """

    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english', device='cpu',
                 batch_size=32, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def embed(self, texts):
        texts = list(map(str, texts))
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc='embedding documents'):
                batch = texts[i:i + self.batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
                ).to(self.device)
                last_hidden = self.model(**encoded).last_hidden_state  # (batch, seq_len, hidden)
                mask = encoded['attention_mask'].unsqueeze(-1).float()  # (batch, seq_len, 1)
                mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings.append(mean_pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)


# ------------------------------------------------------------------------------------
# Feature representations
# ------------------------------------------------------------------------------------
def tfidf_representation(train, tests):
    """
    Vectorizes the training/test collections with TF-IDF. Every method except HistNetQ is
    ultimately built on this representation (HDx/KMM indirectly, via its SVD-reduced projection,
    see low_rank_representation).
    """
    vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(train.X)
    train_numeric = LabelledCollection(Xtr, train.labels, train.classes_)
    tests_numeric = [
        LabelledCollection(vectorizer.transform(test_i.X), test_i.labels, train_numeric.classes_)
        for test_i in tests
    ]
    return train_numeric, tests_numeric


def low_rank_representation(train_numeric, tests_numeric, n_components=5):
    """
    A low-dimensional (TruncatedSVD) projection of a numeric representation, computed once on the
    training collection and reused for every method that wants a dense, low-dimensional feature
    space: HDx (per-feature histogram matching) and KMM (kernel-based) both degrade in high
    dimensions.
    """
    reductor = TruncatedSVD(n_components=n_components, random_state=0)
    train_reduced = LabelledCollection(
        reductor.fit_transform(train_numeric.X), train_numeric.labels, train_numeric.classes_
    )
    tests_reduced = [
        LabelledCollection(reductor.transform(test_i.X), test_i.labels, train_numeric.classes_)
        for test_i in tests_numeric
    ]
    return train_reduced, tests_reduced


def embedded_representation(train, tests):
    """
    Frozen transformer embeddings (see HFEmbedder), used only by HistNetQ. Unlike TF-IDF, these are
    already dense and moderate-dimensional; fed a sparse, high-cardinality TF-IDF matrix instead,
    HistNetQ's per-feature histogram layer would need one histogram per vocabulary entry (an
    impractical parameter count), and the mostly-zero, small-magnitude TF-IDF values would get
    crushed by its Sigmoid into a couple of adjacent bins regardless of document content.
    """
    embedder = HFEmbedder()
    train_embedded = LabelledCollection(embedder.embed(train.X), train.labels, train.classes_)
    tests_embedded = [
        LabelledCollection(embedder.embed(test_i.X), test_i.labels, train.classes_) for test_i in tests
    ]
    return train_embedded, tests_embedded


# ------------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------------
def smooth_curve(dates, values, num_points=300):
    """
    dates: list of timestamps
    values: list of Y-values
    num_points: number of points in the smooth curve

    Returns new_x, new_y for plotting a smooth line.
    """
    # Convert datetime to numeric (matplotlib float representation)
    x = [d.timestamp() for d in dates]
    x = np.array(x)
    y = np.array(values)

    # Create new X-axis with more points
    x_new = np.linspace(x.min(), x.max(), num_points)

    # Smooth spline
    spline = CubicSpline(x, y)
    y_new = spline(x_new)

    # Convert numeric x_new back to datetime
    dates_new = [pd.to_datetime(t, unit='s') for t in x_new]

    return dates_new, y_new


def plot_prevalences(results_dict, target_class=1, target_label='positive', savepath=None):
    """
    Plot prevalence estimates over time for each method contained in results_dict.

    Parameters
    ----------
    results_dict : dict
        A dictionary where:
            - "date-start" : list of datetime-like objects
            - all other keys : list of prevalence vectors (arrays), e.g. [p_pos, p_neg]
              Only the first component (p_pos) will be plotted.
    """
    dates = results_dict["date-start"]

    # Create figure
    plt.figure(figsize=(20, 10))

    # Plot one line per method (except "date-start")
    for method, values in results_dict.items():
        if method == "date-start":
            continue

        # Extract first component from each prevalence array
        target_component = [v[target_class]*100 for v in values]

        dates_smooth, y_smooth = smooth_curve(dates, target_component)

        if method=='true-prev':
            line,=plt.plot(dates_smooth, y_smooth, label=method, linewidth=3, linestyle='-', color='black')
        else:
            line,=plt.plot(dates_smooth, y_smooth, label=method, linewidth=2, linestyle='--')
        plt.plot(dates, target_component, 'o', markersize=10, color=line.get_color())

    # Axis labels
    # plt.xlabel("Date")
    plt.ylabel("% of "+target_label+" tweets")

    # Rotate date labels for readability
    plt.xticks(rotation=45)

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.5)
    plt.grid(which='minor', linestyle=':', linewidth=0.3)

    # Place the legend outside to the right
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if savepath is not None:
        os.makedirs(Path(savepath).parent, exist_ok=True)
        plt.savefig(savepath)
        print(f'plot saved at {savepath}')
    else:
        plt.show()


# ------------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------------

# three representations:
# 'numeric': aggregative quantifiers are trained on a plain LogisticRegression over TFIDF sparse features
#            (see tfidf_representation)
# 'reduced': non-aggregative quantifiers are trained on a SVD-reduced representation of TFIDF features
#            (see low_rank_representation)
# 'embedded': HistNetQ is trained on embedded representations (see embedded_representation)
def methods():
    yield 'CC', CC(LogisticRegression()), 'numeric'
    yield 'ACC', ACC(LogisticRegression()), 'numeric'
    yield 'SLD', EMQ(LogisticRegression()), 'numeric'
    yield 'HDy', DistributionMatchingY(LogisticRegression()), 'numeric'
    yield 'HDx', DistributionMatchingX(), 'reduced'
    yield 'KMM', QUnfoldWrapper(KMM()), 'reduced'
    yield 'KDEy', KDEyML(LogisticRegression()), 'numeric'
    yield 'HistNetQ', HistNetQ(device='cuda'), 'embedded'

train, tests, test_init = prepare_labelled_collections(filter_neutral=True)

train_numeric, tests_numeric = tfidf_representation(train, tests)
train_reduced, tests_reduced = low_rank_representation(train_numeric, tests_numeric, n_components=5)
train_embedded, tests_embedded = embedded_representation(train, tests)

REPRESENTATIONS = {
    'numeric': (train_numeric, tests_numeric),
    'reduced': (train_reduced, tests_reduced),
    'embedded': (train_embedded, tests_embedded),
}

results = defaultdict(list)
for test_i, test_init_i in zip(tests, test_init):
    results['true-prev'].append(test_i.prevalence())
    results['date-start'].append(test_init_i[0])

for q_name, quant, repr_key in methods():
    train_data, test_data = REPRESENTATIONS[repr_key]
    quant.fit(*train_data.Xy)
    for test_i, test_init_i in tqdm(zip(test_data, test_init), desc=f'{q_name} predicting', total=len(test_data)):
        pred_i = quant.predict(test_i.X)
        results[q_name].append(pred_i)

plot_prevalences(results, savepath='./plots/over_time.pdf')
print('[done]')



