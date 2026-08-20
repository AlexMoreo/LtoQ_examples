import itertools

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



palette = itertools.cycle(sns.color_palette())

def setframe():
    fig.spines['top'].set_visible(False)
    fig.spines['left'].set_visible(False)
    fig.get_yaxis().set_ticks([])
    fig.spines['right'].set_visible(False)
    # fig.axis('off')

nbins = 50
figsize = (5, 2)
ymax = 0.2

negatives = np.random.normal(loc = 0.3, scale=0.2, size=20000)
negatives = np.asarray([x for x in negatives if 0 <= x <= 1])

plt.figure(figsize=figsize)
plt.xlim(0, 1)
plt.ylim(0, ymax)
fig = sns.histplot(data=negatives, binrange=(0,1), bins=nbins,  stat='probability', color=next(palette))
plt.title('Negative distribution')
fig.set(yticklabels=[])
fig.set(ylabel=None)
setframe()
# fig.get_figure().savefig('plots_cacm/negatives.pdf')
# plt.clf()

# -------------------------------------------------------------

positives1 = np.random.normal(loc = 0.75, scale=0.06, size=20000)
positives2 = np.random.normal(loc = 0.65, scale=0.1, size=1)
positives = np.concatenate([positives1, positives2])
np.random.shuffle(positives)
positives = np.asarray([x for x in positives if 0 <= x <= 1])

# plt.figure(figsize=figsize)
plt.xlim(0, 1)
plt.ylim(0, ymax)
fig = sns.histplot(data=positives, binrange=(0,1), bins=nbins,  stat='probability', color=next(palette))
plt.title('')
fig.set(yticklabels=[])
fig.set(ylabel=None)
setframe()
fig.get_figure().savefig('plots/training.pdf')

# -------------------------------------------------------------

prev = 0.2
test = np.concatenate([
    negatives[:int(len(negatives)*(1-prev))],
    positives[:int(len(positives)*(prev))],
])


plt.figure(figsize=figsize)
plt.xlim(0, 1)
plt.ylim(0, ymax)
fig = sns.histplot(data=test, binrange=(0,1), bins=nbins,  stat='probability', color=next(palette))
plt.title('')
fig.set(yticklabels=[])
fig.set(ylabel=None)
setframe()
fig.get_figure().savefig('plots/test.pdf')


