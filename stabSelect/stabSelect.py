__author__ = 'yuxinsun'

import numpy as np
import scipy.sparse as sp
from opt import fitAlg


class StabSelect():
    """ Stability selection with a range of feature selection algorithms

    Parameters:

    -------------------------
    global:
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
            for LPBoost and exclusive group Lasso, installation of [??] and [??] is required

    :param n_iter: int
        number of iterations in stability selection

    :param p_threshold: float
        threshold for selection probabilities in stability selection
        this parameter does not necessarily need to be predefined as the complete selection probability matrix will be returned

    :param sample_size: int
        the number of samples required in the subsampling process in stability selection
        if not defined, half of the samples will be used

    :param verbose: int, either 0 or 1
        if verbose = 1, log of optimisation processes will be printed out

    -------------------------
    algorithm-specific:
    :param param_range: list, length n_parameters
        list of regularisation parameters
        Lasso, randomised Lasso, and exclusive group Lasso: list of lambda values
        LPBoost: list of nu values
        elastic net: list of paired alpha and l1_ratio

    :param alpha: float
        alpha in randomised Lasso

    :param idx_group: array-like, shape: [??]
        indicator matrix of group allocation in exclusive group Lasso
        does require predefinition if n_group is specified

    :param n_group: int
        the number of group in exclusive group Lasso
        if idx_group is not defined, n_group random groups will be created


    Return (as attributes):
    -------------------------
    :return idx: array-like, shape (n_select_feature, )
        indiced of selected features, using the specified threshold

    :return select_prob: array-like, shape (n_feature, n_parameter)
        complete matrix of selection probabilities

    :return weights_total: list, length n_iter
        estimated weights from individual iterations in stability selection

    :return max_prob: array-like, shape (n_features, )
        maximum selection probability over all parameters, not recommended for use

    :return convergence: list, length n_iter
        list of boolean variables that indicate convergence of individual iterations of stability selection

    """

    def __init__(self, param_range=np.linspace(0.01, 1, 100), alpha=0.5, l1_ratio=0.5, sample_size=None, p_threshold=0.5, n_iter=100,
                 reg_type='lasso', verbose=0, idx_group=None, n_group=50):
        self.param_range = param_range
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.sample_size = sample_size
        self.p_threshold = p_threshold
        self.n_iter = n_iter
        self.reg_type = reg_type
        self.verbose = verbose
        self.idx_group = idx_group
        self.n_group = n_group

    def _compute_w(self, X, y):
        """
        Estimate weights/coefficients with specified selection algorithm

        :param X: array-like, shape (n_subsample, n_feature)
            subsampled input features

        :param y: array-like, shape (n_subsample, )
            subsampled input labels

        :return w_vec: array-like, shape (n_feature, n_parameter)
            estimated weights over all parameters under current iteration

        """
        return fitAlg(X, y, param_range=self.param_range, alpha=self.alpha, reg_type=self.reg_type,
                        idx_group=self.idx_group, n_group=self.n_group)

    def _select_feature(self, w_vec):
        """
        Select features with absolute weights above a threshold (10**-3, can be modified)

        :param w_vec: array-like, shape (n_feature, n_parameter)
            estimated weights over all parameters under current iteration

        :return: array-like, shape (n_feature, n_parameter)
            indicator matrix of selected features, selected: 1, not selected: 0

        """
        w_vec_new = sp.lil_matrix(w_vec.shape)
        w_vec_new[np.where(abs(w_vec) >= 10**-3)] = 1.

        return sp.csr_matrix(w_vec_new)

    def _get_weights(self, w_vec):
        """
        Obtain indices of selected features

        :param w_vec: array-like, shape (n_feature, n_parameter)
            estimated weights over all parameters under current iteration

        :return idx: array_like, shape (n_selected_feature, )  [??]
            indices of selected features

        """
        idx = np.where(abs(w_vec) >= 10**-3)

        return idx[0]

    def _fit(self, X, y):

        n_sample, n_feature = X.shape
        if self.sample_size is None:
            self.sample_size = int(n_sample / 2)

        w_counter = sp.csr_matrix((n_feature, len(self.param_range)))
        w_vals, idx_vals, convergence = [], [], []

        # stability selection
        for counter in range(self.n_iter):

            if self.verbose == 1:
                print('iteration: %d/%d.' % (counter, self.n_iter))

            # subsample samples
            sub_idx = np.random.permutation(n_sample)[:self.sample_size]

            X_sub = X[sub_idx, :]
            y_sub = y[sub_idx]

            # fit subsamples to get coefficients
            w_vec, converged = self._compute_w(X_sub, y_sub)
            if converged:
                w_counter += self._select_feature(w_vec)

            idx_temp = self._get_weights(w_vec)
            w_vals.append(w_vec)
            idx_vals.append(idx_temp)
            convergence.append(converged)

        # compute selection probabilities
        # select features with selection probabilities larger than p_threshold under all lambdas
        select_prob = w_counter / float(len(np.where(convergence)[0]))
        max_select_prob = np.max(select_prob, axis=1)
        idx = np.where(max_select_prob >= self.p_threshold)[0]

        self.idx = idx
        self.w_count = w_counter
        self.select_prob = select_prob
        self.max_prob = max_select_prob
        self.weights_total = w_vals
        self.idx_total = idx_vals
        self.convergence = convergence

    def fit(self, X, y):
        """
        Fit stability selection

        :param X: array-like, shape (n_subsample, n_feature)
            subsampled input features
        :param y: array-like, shape (n_subsample, )
            subsampled input labels
        """
        self._fit(X, y)
