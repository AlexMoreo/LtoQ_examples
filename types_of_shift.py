import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde_background(ax, data, cmap="Blues", alpha=0.35, gridsize=200):
    """
    data: array Nx2
    """
    # KDE
    kde = gaussian_kde(data.T)

    # Grid for evaluation
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, gridsize),
        np.linspace(y_min, y_max, gridsize)
    )
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Draw background density
    ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=alpha)


# ======================================================
#  Define 3 Gaussian sources in 2D
# ======================================================

# Means
mu1 = np.array([0, 0])  # negative
mu2 = np.array([3, 0])  # positive
mu3 = np.array([0, 3])  # positive

# Covariances
Sigma = np.array([[1, 0.2],
                  [0.2, 1]])


def sample_gaussian(mu, Sigma, n):
    return np.random.multivariate_normal(mu, Sigma, n)


# ======================================================
#  Generate datasets for the 4 scenarios
# ======================================================

density = 20

# ---------- Scenario 1: Baseline ----------
G1_1 = sample_gaussian(mu1, Sigma, 100*density)
G2_1 = sample_gaussian(mu2, Sigma, 100*density)
G3_1 = sample_gaussian(mu3, Sigma, 100*density)

# ---------- Scenario 2: Prior Probability Shift ----------
G1_2 = sample_gaussian(mu1, Sigma, 300*density)
G2_2 = sample_gaussian(mu2, Sigma, 50*density)
G3_2 = sample_gaussian(mu3, Sigma, 50*density)

# ---------- Scenario 3: Covariate Shift ----------
# same class proportions but G3 moves (X-shift)
mu3_shift = mu3 + np.array([1.5, 0])
G1_3 = sample_gaussian(mu1, Sigma, 100*density)
G2_3 = sample_gaussian(mu2, Sigma, 100*density)
G3_3 = sample_gaussian(mu3_shift, Sigma, 100*density)  # shifted covariates

# ---------- Scenario 4: Concept Shift ----------
# same data as Scenario 1, but G3 becomes negative
G1_4 = G1_1
G2_4 = G2_1
G3_4 = G3_1  # but will be colored as negative


# ======================================================
#  Plotting function for each scenario
# ======================================================

def plot_scenario(ax, G1, G2, G3, title, G3_negative=False):
    # plot_kde_background(ax, G1, cmap="Reds", alpha=0.75)
    # plot_kde_background(ax, G2, cmap="Blues", alpha=0.75)
    # plot_kde_background(ax, G3, cmap="Greens", alpha=0.75)

    ax.scatter(G1[:, 0], G1[:, 1], s=12, color='red', alpha=0.1, label='Negative ($\ominus$)')
    ax.scatter(G2[:, 0], G2[:, 1], s=12, color='blue', alpha=0.1, label='Positive ($\oplus$)')

    if G3_negative:
        ax.scatter(G3[:, 0], G3[:, 1], s=12, color='red', alpha=0.1) #, label='Negative ($\ominus$)')
    else:
        ax.scatter(G3[:, 0], G3[:, 1], s=12, color='blue', alpha=0.1) #, label='Positive ($\oplus$)')

    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.3)


# ======================================================
#  Generate 2×2 grid of subplots
# ======================================================

fig, axes = plt.subplots(2, 2, figsize=(9, 9))

plot_scenario(axes[0, 0], G1_1, G2_1, G3_1,
              "Training data")

plot_scenario(axes[0, 1], G1_2, G2_2, G3_2,
              "Prior Probability Shift")

plot_scenario(axes[1, 0], G1_3, G2_3, G3_3,
              "Covariate Shift",
              G3_negative=False)

plot_scenario(axes[1, 1], G1_4, G2_4, G3_4,
              "Concept Shift",
              G3_negative=True)

# One global legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
plt.savefig('dataset_shift_types.pdf')
