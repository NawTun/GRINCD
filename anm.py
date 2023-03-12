from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


def rbf_dot(X, deg):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H

# use HSIC to evaluate independence of two variables X and Y
def FastHsicTestGamma(X, Y, sig=[-1, -1], maxpnt=200):
    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        #       indx = np.r_[0:maxpnt]
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(Xm, sig[0])
    L = rbf_dot(Ym, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0
    return testStat


def normalized_hsic(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    h = FastHsicTestGamma(x, y)

    return h


class ANM:

    def __init__(self):
        super(ANM, self).__init__()

    def predict_proba(self, data, **kwargs):
        """
        Prediction method for pairwise causal inference using the ANM model.
        :param data: Couple of np.ndarray variables to classify.
        :return: Causation score.
        """
        a, b = data
        a_normal = (a - a.mean()) / a.std()
        b_normal = (b - b.mean()) / b.std()
        a = a_normal.reshape((-1, 1))
        b = b_normal.reshape((-1, 1))
        return self.anm_score(b, a) - self.anm_score(a, b)

    def anm_score(self, x, y):
        """
        Compute the fitness score of the ANM model in the x->y direction.
        :param x: Variable seen as cause.
        :param y: Variable seen as effect.
        :return: ANM fit score.
        """
        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        indepscore = normalized_hsic(y_predict - y, x)
        return indepscore
