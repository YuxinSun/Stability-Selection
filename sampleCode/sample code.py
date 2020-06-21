__author__ = 'yuxinsun'

import numpy as np
from exclGroupLasso.ExclGroupLasso import reweightEG
from stabSelect.stabSelect import StabSelect
from sklearn.linear_model import Lasso


def printResult(idx, idx_relevant, n_relevant, algorithm):
    print '\n%s\n' \
          '# Selected features: %d, # Selected relevant features: %d.\n' \
          'Precision: %.2f, recall: %.2f.' \
          % (algorithm, len(idx), len(idx_relevant),
             len(idx_relevant) / float(len(idx)), len(idx_relevant) / float(n_relevant))


# create synthetic data
n_sample, n_feature, n_relevant = 500, 100, 30
n_trn = 400

X = np.random.randn(n_sample, n_feature)
w = np.random.randn(n_relevant)
n = np.random.randn(n_sample)
y = X[:, :n_relevant].dot(w) + n

X_trn, X_tst = X[:n_trn, :], X[n_trn:, :]
y_trn, y_tst = y[:n_trn], y[n_trn:]


# exclusive group lasso with random allocated groups
reg = reweightEG(alpha=2**10, n_group=50)
reg.fit(X_trn, y_trn)

idx = reg.idx  # indices of selected features
idx_relevant = set(idx).intersection(set(range(n_relevant)))

printResult(idx, idx_relevant, n_relevant, 'Exclusive group Lasso')


# exclusive group lasso with stability selection + random allocated groups
reg = StabSelect(param_range=[2**5], reg_type='excl grp lasso', p_threshold=0.9, n_group=50, n_iter=50, verbose=0)
reg.fit(X_trn, y_trn)

idx = reg.idx
idx_relevant = set(idx).intersection(set(range(n_relevant)))

printResult(idx, idx_relevant, n_relevant, 'Exclusive group Lasso with stability selection')


# lasso
reg = Lasso(alpha=2**-3, fit_intercept=False)
reg.fit(X_trn, y_trn)

idx = np.where(reg.coef_ != 0)[0]
idx_relevant = set(idx).intersection(set(range(n_relevant)))

printResult(idx, idx_relevant, n_relevant, 'Lasso')

