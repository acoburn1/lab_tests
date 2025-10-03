import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import eigh

def pca_embedding(A_rows, target=0.95):
    # Center the data
    Xc = A_rows - A_rows.mean(axis=0)
    # Covariance matrix
    C = np.cov(Xc, rowvar=False)
    # Eigendecomposition
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    # Projection
    X_proj = Xc @ evecs
    var_ratio = evals / evals.sum()
    cume = np.cumsum(var_ratio)
    k95 = np.searchsorted(cume, target) + 1
    return X_proj, evals, var_ratio, cume, k95


def get_pcns_mod_lat(activations, num_features):
    _, _, _, _, km = pca_embedding(activations[:num_features])
    _, _, _, _, kl = pca_embedding(activations[num_features:])
    return km, kl
    