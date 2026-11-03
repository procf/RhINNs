"""Sequential threshold ridge regression (Rudy et al., Sci. Adv. 2017)."""
import numpy as np


def train_stridge(R0, Ut, lam, d_tol, w_init, maxit=10, str_iters=100,
                  l0_penalty=None, normalize=2, split=0.8):
    """Sparse solution of ``Ut = R0 @ w`` with an adaptively chosen threshold.

    The columns of ``R0`` are normalised, a random ``split`` of the rows is
    used for fitting and the rest to score each threshold
    (``||Ut - R0 w||_2 + l0_penalty * nnz(w)``). The threshold grows by
    ``d_tol`` while the score improves and backs off otherwise.
    ``w_init`` (the current coefficients) seeds the first STRidge pass.
    """
    np.random.seed(0)
    n, d = R0.shape
    R = np.zeros((n, d), dtype=np.float32)
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(d):
            Mreg[i] = 1.0 / np.linalg.norm(R0[:, i], normalize)
            R[:, i] = Mreg[i] * R0[:, i]
        normalize_inner = 0
    else:
        R = R0
        Mreg = np.ones((d, 1)) * d
        normalize_inner = 2

    train = np.random.choice(n, int(n * split), replace=False)
    test = np.setdiff1d(np.arange(n), train)
    TrainR, TestR = R[train, :], R[test, :]
    TrainY, TestY = Ut[train, :], Ut[test, :]

    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty is None:
        l0_penalty = 0.001 * np.linalg.cond(R)

    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]
    err_best = (np.linalg.norm(TestY - TestR.dot(w_best), 2)
                + l0_penalty * np.count_nonzero(w_best))

    for it in range(maxit):
        w = stridge(TrainR, TrainY, lam, str_iters, tol, Mreg, w_init,
                    normalize=normalize_inner)
        err = (np.linalg.norm(TestY - TestR.dot(w), 2)
               + l0_penalty * np.count_nonzero(w))
        if err <= err_best:
            err_best, w_best = err, w
            tol = tol + d_tol
        else:
            tol = max(0, tol - 2 * d_tol)
            d_tol = 2 * d_tol / (maxit - it)
            tol = tol + d_tol

    return np.real(np.multiply(Mreg, w_best))


def stridge(X0, y, lam, maxit, tol, Mreg, w_init, normalize=2):
    """Ridge regression, repeatedly zeroing coefficients smaller than ``tol``."""
    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(d):
            Mreg[i] = 1.0 / np.linalg.norm(X0[:, i], normalize)
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Start from the current coefficients expressed in the normalised basis.
    w = w_init / Mreg
    biginds = np.where(abs(w) > tol)[0]
    num_relevant = d

    for j in range(maxit):
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)

        if len(new_biginds) == 0:
            if j == 0:
                return np.multiply(Mreg, w) if normalize != 0 else w
            break
        biginds = new_biginds
        w[smallinds] = 0

        if lam != 0:
            w[biginds] = np.linalg.lstsq(
                X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
                X[:, biginds].T.dot(y), rcond=None)[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if len(biginds) > 0:
        w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    return np.multiply(Mreg, w) if normalize != 0 else w
