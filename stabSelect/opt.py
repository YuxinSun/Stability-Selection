__author__ = 'yuxinsun'

import numpy as np
from sklearn.linear_model import Lasso, ElasticNet, SGDClassifier

"""


"""


def standard_lasso(X, y, param_val):
    clf = Lasso(alpha=param_val, fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_), True


def random_lasso(X, y, param_val, alpha):
    n_sample, n_feature = X.shape

    # random re-scale X
    # W = np.random.random_sample(n_feature)
    # W = (1. - alpha) * W + alpha
    W = np.random.uniform(alpha, 1, n_feature)
    X *= W[None, :]

    w_tran = standard_lasso(X, y, param_val)
    w = np.multiply(w_tran, W)

    return w, True


def elastic_net(X, y, param_val):
    clf = ElasticNet(alpha=param_val[0], l1_ratio=param_val[1], fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_), True


def elastic_logistic(X, y, param_val):
    clf = SGDClassifier(loss='log', penalty='elasticnet', alpha=param_val[0], l1_ratio=param_val[1], fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_), True


def elastic_hinge(X, y, param_val):
    clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=param_val[0], l1_ratio=param_val[1], fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_), True


def l1_hinge(X, y, param_val):
    clf = SGDClassifier(loss='hinge', penalty='l1', alpha=param_val, fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_), True


def lp(X, y, param_val):
    clf = lpboost(nu=param_val)
    clf.fit(X, y)

    w = np.zeros(X.shape[1])
    w[clf.idx] = clf.a

    return w, True


def l12_norm(X, y, idx_group, n_group, param_val):

    if idx_group is None:  # random group
        n_feature = X.shape[1]

        idx_group = np.zeros((n_group, n_feature))
        idx = np.random.permutation(n_feature)
        idx = np.array_split(idx, n_group)

        for sub_counter, sub_idx in enumerate(idx):
            idx_group[sub_counter, sub_idx] = 1


    else:  # random fixed group: fix useful features are uniformly distributed across groups,
           # other features within each group are randomly chosen

        n_group, n_feature = idx_group.shape

        if np.all(np.equal(np.diag(idx_group[:, -n_group:]), np.ones(n_group))):  # artificial features
            print 'artificial'

            n_feature_X = n_feature - n_group  # n_original_feature = n_art_feature - n_group
            idx_art = idx_group[:, -n_group:]
            idx = np.random.permutation(n_feature_X)

            idx_group = idx_group[:, :n_feature_X]
            idx_group = np.hstack((idx_group[:, idx], idx_art))

        elif np.array_equal(np.nonzero(idx_group[:, :n_group])[1], range(n_group)):  # fixed group
            print 'fixed'

            idx = np.random.permutation(n_feature - n_group)  # for fixed group, n_group = n_useful_feature
            idx = np.array_split(idx, n_group)

            idx_group = np.zeros((n_group, n_feature - n_group))
            for sub_counter, sub_idx in enumerate(idx):
                idx_group[sub_counter, sub_idx] = 1

            idx_group = np.hstack((np.eye(n_group), idx_group))

        else:  # customised group with no need to change
            pass

    clf = l12_norm_transformed(c=param_val, idx_group=idx_group)
    clf.fit(X, y)

    return clf.coef, clf.converged


def l12_sparse(X, y, idx_group, n_group, param_val, random_state=None):

    if idx_group is None:  # random group
        n_feature = X.shape[1]

        # idx_group = np.zeros((n_group, n_feature))
        idx_group = sp.lil_matrix((n_group, n_feature))  # modified for sparse
        idx = np.random.permutation(n_feature)
        idx = np.array_split(idx, n_group)

        for sub_counter, sub_idx in enumerate(idx):
            idx_group[sub_counter, sub_idx] = 1

        idx_group = sp.csr_matrix(idx_group)  # modified for sparse

    else:  # random fixed group: fix useful features are uniformly distributed across groups,
           # other features within each group are randomly chosen

        n_group, n_feature = idx_group.shape

        if np.all(np.equal(np.diag(idx_group[:, -n_group:]), np.ones(n_group))):  # artificial features
            print 'artificial'

            n_feature_X = n_feature - n_group  # n_original_feature = n_art_feature - n_group
            idx_art = idx_group[:, -n_group:]
            idx = np.random.permutation(n_feature_X)

            idx_group = idx_group[:, :n_feature_X]
            idx_group = np.hstack((idx_group[:, idx], idx_art))

        if np.array_equal(np.nonzero(idx_group[:, :n_group])[1], range(n_group)):  # fixed group

            idx = np.random.permutation(n_feature - n_group)  # for fixed group, n_group = n_useful_feature
            idx = np.array_split(idx, n_group)

            idx_group = np.zeros((n_group, n_feature - n_group))
            for sub_counter, sub_idx in enumerate(idx):
                idx_group[sub_counter, sub_idx] = 1

            idx_group = np.hstack((np.eye(n_group), idx_group))

        else:  # customised group with no need to change
            pass

    clf = l12_norm_sparse(c=param_val, idx_group=idx_group, verbose=1)
    clf.fit(X, y)

    return clf.coef, clf.converged

def fitAlg(X, y, param_range, alpha, reg_type, idx_group=None, n_group=None):
    n_sample, n_feature = X.shape
    w_vec = np.empty((n_feature, len(param_range)))

    if reg_type == 'lasso':
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = standard_lasso(X, y, param_val)
    elif reg_type =='random lasso':
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = random_lasso(X, y, param_val, alpha)
    elif reg_type == 'elastic net':
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = elastic_net(X, y, param_val)
    elif reg_type == 'elastic logistic':  # elastic net with logistic loss
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = elastic_logistic(X, y, param_val)
    elif reg_type == 'elastic hinge':  # elastic net with hinge loss
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = elastic_hinge(X, y, param_val)
    elif reg_type == 'l1 hinge':  # l1 regularised classification with hinge loss
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = l1_hinge(X, y, param_val)
    elif reg_type == 'lpboost':  # lpboost
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter], converged = lp(X, y, param_val)
    elif reg_type == 'l12_norm':
        if idx_group is None and n_group is None:
            raise KeyError('must specify at least one between idx_group and n_group')
        for counter, param_val in enumerate(param_range):
            print 'param(%d/%d): %.4f' % (counter, len(param_range), param_val)
            w_vec[:, counter], converged = l12_norm(X, y, idx_group, n_group, param_val)
    elif reg_type == 'l12_norm_sparse':
        if idx_group is None and n_group is None:
            raise KeyError('must specify at least one between idx_group and n_group')
        for counter, param_val in enumerate(param_range):
            print 'param(%d/%d): %.4f' % (counter, len(param_range), param_val)
            w_vec[:, counter], converged = l12_sparse(X, y, idx_group, n_group, param_val)
    else:
        raise KeyError('not implemented yet')

    return w_vec, converged