__author__ = 'yuxinsun'

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Lasso, ElasticNet, SGDClassifier
from exclGroupLasso.ExclGroupLasso import reweightEG
# from LPBoost.lpboost import lpboost
# from exclGroupLasso.ExclGroupLasso import l12_norm_transformed, l12_norm_sparse


def standard_lasso(X, y, param_val):
    """
    Lasso

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param param_val: float
        regularisation parameter lambda in Lasso

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    clf = Lasso(alpha=param_val, fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_)


def random_lasso(X, y, param_val, alpha):
    """
    Randomised Lasso

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param param_val: float
        regularisation parameter lambda in Lasso
    :param alpha: float
        lower bound of reweighted features in randomised Lasso, alpha \in (0, 1]

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    if alpha > 1 or alpha <= 0:
        raise KeyError('alpha must be in (0, 1] in randomised Lasso.')

    n_sample, n_feature = X.shape

    W = np.random.uniform(alpha, 1, n_feature)
    X *= W[None, :]  # [??]

    w_tran = standard_lasso(X, y, param_val)
    w = np.multiply(w_tran, W)

    return w


def elastic_net(X, y, param_val):
    """
    Elastic net

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param param_val: tuple or list, length 2
        regularisation parameter alpha and l1_ratio in elastic net

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    clf = ElasticNet(alpha=param_val[0], l1_ratio=param_val[1], fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_)


def elastic_logistic(X, y, param_val):
    """
    Elastic net with logistic loss

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param param_val: tuple or list, length 2
        regularisation parameter alpha and l1_ratio in elastic net

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    clf = SGDClassifier(loss='log', penalty='elasticnet', alpha=param_val[0], l1_ratio=param_val[1], fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_)


def elastic_hinge(X, y, param_val):
    """
    Elastic net with hinge loss

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param param_val: tuple or list, length 2
        regularisation parameter alpha and l1_ratio in elastic net

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=param_val[0], l1_ratio=param_val[1], fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_)


def l1_hinge(X, y, param_val):
    """
    l1-regularised SVM with linear kernel

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param param_val: float
        regularisation parameter in SVM

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    clf = SGDClassifier(loss='hinge', penalty='l1', alpha=param_val, fit_intercept=False)
    clf.fit(X, y)

    return np.ravel(clf.coef_)


# def lp(X, y, param_val):
#     """
#     LPBoost
#
#     -------------------------
#     :param X: array-like, shape (n_sample, n_feature)
#         input features
#     :param y: array-like, shape (n_sample, )
#         input labels
#     :param param_val: float
#         regularisation parameter nu in LPBoost
#
#     -------------------------
#     :return: array-like, shape (n_feature, )
#         estimated weights/coefficients
#     """
#
#     clf = lpboost(nu=param_val)
#     clf.fit(X, y)
#
#     w = np.zeros(X.shape[1])
#     w[clf.idx] = clf.a
#
#     return w


def l12_norm(X, y, idx_group, n_group, param_val):
    """
    Exclusive group Lasso

    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features
    :param y: array-like, shape (n_sample, )
        input labels
    :param idx_group: array-like, shape (n_group, n_feature)
        indicator matrix of group allocation
    :param n_group: int
        number of groups, must be specified if idx_group is not defined
        when idx_group is not defined, n_group random groups will be created
    :param param_val: tuple or list, length 2
        regularisation parameter lambda in exclusive group Lasso

    -------------------------
    :return: array-like, shape (n_feature, )
        estimated weights/coefficients
    """

    if idx_group is None:  # random group

        if n_group is None:
            raise KeyError('n_group must be specified when idx_group is None.')
        elif not (isinstance(n_group, int) or (isinstance(n_group, float) and n_group.is_integer())):
            raise KeyError('n_group must be an integer.')

        n_feature = X.shape[1]

        idx_group = np.zeros((n_group, n_feature))
        idx = np.random.permutation(n_feature)
        idx = np.array_split(idx, n_group)

        for sub_counter, sub_idx in enumerate(idx):
            idx_group[sub_counter, sub_idx] = 1


    else:

        n_group, n_feature = idx_group.shape

        if np.all(np.equal(np.diag(idx_group[:, -n_group:]), np.ones(n_group))):  # artificial features

            n_feature_X = n_feature - n_group  # n_original_feature = n_art_feature - n_group
            idx_art = idx_group[:, -n_group:]
            idx = np.random.permutation(n_feature_X)

            idx_group = idx_group[:, :n_feature_X]
            idx_group = np.hstack((idx_group[:, idx], idx_art))

        elif np.array_equal(np.nonzero(idx_group[:, :n_group])[1], range(n_group)):  # fixed group

            idx = np.random.permutation(n_feature - n_group)  # for fixed group, n_group = n_useful_feature
            idx = np.array_split(idx, n_group)

            idx_group = np.zeros((n_group, n_feature - n_group))
            for sub_counter, sub_idx in enumerate(idx):
                idx_group[sub_counter, sub_idx] = 1

            idx_group = np.hstack((np.eye(n_group), idx_group))

        else:  # customised group with no need to change
            pass

    clf = reweightEG(alpha=param_val, idx_group=idx_group)
    clf.fit(X, y)

    return clf.coef


# def l12_sparse(X, y, idx_group, n_group, param_val, random_state=None):
#     """
#     Exclusive group Lasso, optimised for sparse matrices
#
#     -------------------------
#     :param X: array-like, shape (n_sample, n_feature)
#         input features
#     :param y: array-like, shape (n_sample, )
#         input labels
#     :param idx_group: array-like, shape (n_group, n_feature)
#         indicator matrix of group allocation
#     :param n_group: int
#         number of groups, must be specified if idx_group is not defined
#         when idx_group is not defined, n_group random groups will be created
#     :param param_val: tuple or list, length 2
#         regularisation parameter lambda in exclusive group Lasso
#
#     -------------------------
#     :return: array-like, shape (n_feature, )
#         estimated weights/coefficients
#     """
#
#     if idx_group is None:  # random group
#
#         if n_group is None:
#             raise KeyError('n_group must be specified when idx_group is None.')
#         elif not (isinstance(n_group, int) or (isinstance(n_group, float) and n_group.is_integer())):
#             raise KeyError('n_group must be an integer.')
#
#         n_feature = X.shape[1]
#
#         # idx_group = np.zeros((n_group, n_feature))
#         idx_group = sp.lil_matrix((n_group, n_feature))  # modified for sparse
#         idx = np.random.permutation(n_feature)
#         idx = np.array_split(idx, n_group)
#
#         for sub_counter, sub_idx in enumerate(idx):
#             idx_group[sub_counter, sub_idx] = 1
#
#         idx_group = sp.csr_matrix(idx_group)  # modified for sparse
#
#     else:
#
#         n_group, n_feature = idx_group.shape
#
#         if np.all(np.equal(np.diag(idx_group[:, -n_group:]), np.ones(n_group))):  # artificial features
#             print 'artificial'
#
#             n_feature_X = n_feature - n_group  # n_original_feature = n_art_feature - n_group
#             idx_art = idx_group[:, -n_group:]
#             idx = np.random.permutation(n_feature_X)
#
#             idx_group = idx_group[:, :n_feature_X]
#             idx_group = np.hstack((idx_group[:, idx], idx_art))
#
#         if np.array_equal(np.nonzero(idx_group[:, :n_group])[1], range(n_group)):  # fixed group
#
#             idx = np.random.permutation(n_feature - n_group)  # for fixed group, n_group = n_useful_feature
#             idx = np.array_split(idx, n_group)
#
#             idx_group = np.zeros((n_group, n_feature - n_group))
#             for sub_counter, sub_idx in enumerate(idx):
#                 idx_group[sub_counter, sub_idx] = 1
#
#             idx_group = np.hstack((np.eye(n_group), idx_group))
#
#         else:  # customised group with no need to change
#             pass
#
#     clf = l12_norm_sparse(c=param_val, idx_group=idx_group, verbose=1)
#     clf.fit(X, y)
#
#     return clf.coef


def fitAlg(X, y, param_range, alpha, reg_type, idx_group=None, n_group=None, verbose=0):
    """
    Fit feature selection algorithms

    Parameters:
    -------------------------
    :param X: array-like, shape (n_sample, n_feature)
        input features

    :param y: array-like, shape (n_sample, )
        input labels

    :param param_range: list, length n_parameters
        list of regularisation parameters
        Lasso, randomised Lasso, and exclusive group Lasso: list of lambda values
        LPBoost: list of nu values
        elastic net: list of paired alpha and l1_ratio

    :param alpha: float
        alpha in randomised Lasso

    :param reg_type: string
        selection algorithm, chosen from:
        lasso: Lasso (solved with coordinate descent)
        random lasso: randomised Lasso  (solved with coordinate descent)
        elastic net: elastic net with square loss (solved with coordinate descent?)
        elastic logistic: elastic net with logistic loss (solved with stochastic gradient descent)
        elastic hinge: elastic net with hinge loss (solved with stochastic gradient descent)
        l1 hinge: l1-regularised svm (solved with stochastic gradient descent)
        lpboost: LPBoost (almost equivalent to l1-regularised svm, only returned postive weights, solved by linear programming)
        excl lasso: exclusive group Lasso (solved by multiple implementations)
        excl lasso sparse: exclusive group Lasso (optimised for sparse feature matrices, solved by multiple implementations)
        for LPBoost and exclusive group Lasso, installation of LPBoost and exclGroupLasso is required

    :param idx_group:  array-like, shape (n_group, n_feature)
        indicator matrix of group allocation in exclusive group Lasso
        does require predefinition if n_group is specified

    :param n_group: int
        the number of group in exclusive group Lasso
        if idx_group is not defined, n_group random groups will be created


    Return (as attributes):
    -------------------------
    :return w_vec: array-like, shape (n_feature, n_parameter)
        estimated weights over all parameters under current iteration

    :return converged: list, length n_parameters
        list of boolean variables indicating convergence under each parameter
    """

    n_sample, n_feature = X.shape
    w_vec = np.empty((n_feature, len(param_range)))

    if reg_type == 'lasso':  # Lasso
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = standard_lasso(X, y, param_val)
    elif reg_type =='random lasso':  # randomised Lasso
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = random_lasso(X, y, param_val, alpha)
    elif reg_type == 'elastic net':  # elastic net
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = elastic_net(X, y, param_val)
    elif reg_type == 'elastic logistic':  # elastic net with logistic loss
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = elastic_logistic(X, y, param_val)
    elif reg_type == 'elastic hinge':  # elastic net with hinge loss
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = elastic_hinge(X, y, param_val)
    elif reg_type == 'l1 hinge':  # l1 regularised classification with hinge loss
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = l1_hinge(X, y, param_val)
    elif reg_type == 'lpboost':  # lpboost
        for counter, param_val in enumerate(param_range):
            w_vec[:, counter] = lp(X, y, param_val)
    elif reg_type == 'excl grp lasso':  # exclusive group Lasso
        if idx_group is None and n_group is None:
            raise KeyError('must specify at least one between idx_group and n_group')
        for counter, param_val in enumerate(param_range):
            if verbose == 1:
                print 'param(%d/%d): %.4f' % (counter, len(param_range), param_val)
            w_vec[:, counter] = l12_norm(X, y, idx_group, n_group, param_val)
    elif reg_type == 'l12_norm_sparse':  # exclusive group Lasso, optimised for sparse matrices
        if idx_group is None and n_group is None:
            raise KeyError('must specify at least one between idx_group and n_group')
        for counter, param_val in enumerate(param_range):
            if verbose == 1:
                print 'param(%d/%d): %.4f' % (counter, len(param_range), param_val)
            w_vec[:, counter] = l12_sparse(X, y, idx_group, n_group, param_val)
    else:  # other algorithms are not implemented
        raise KeyError('not implemented yet')

    return w_vec