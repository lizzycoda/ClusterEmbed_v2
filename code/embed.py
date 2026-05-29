"""
Additional embedding method not readily available in sklearn.
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path


class LandmarkIsomap:
    def __init__(self, n_components=2, n_landmarks=300, random_state=0):
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.random_state = random_state

    def fit_transform(self, adata):
        graph = adata.obsp["distances"]  # kNN graph from Scanpy
        rng = np.random.RandomState(self.random_state)
        N = graph.shape[0]

        # 1. pick landmarks
        landmarks = rng.choice(N, self.n_landmarks, replace=False)
        self.landmarks_ = landmarks

        # 2. compute distances: landmarks to remaining nodes 
        D_LN = shortest_path(
            graph,
            directed=False,
            indices=landmarks
        )
        D_LN[np.isinf(D_LN)] =  np.max(D_LN[np.isfinite(D_LN)]) 
        D_NL = D_LN.T

        # 3. landmark-landmark distances
        D_LL = D_NL[landmarks]

        # 4. classical MDS on landmarks
        D2_LL = D_LL ** 2
        n = D_LL.shape[0]

        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2_LL @ H

        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1][:self.n_components]

        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # landmark embedding
        L_inv = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12)))

        D2 = D_NL ** 2

        row_mean = D2.mean(axis=1, keepdims=True)
        col_mean = D2.mean(axis=0, keepdims=True)
        total_mean = D2.mean()

        B_ext = -0.5 * (D2 - row_mean - col_mean + total_mean)

        # full embedding
        Y = B_ext @ eigvecs @ L_inv

        return Y