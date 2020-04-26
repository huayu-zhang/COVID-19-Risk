from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from statsmodels.formula.api import logit
from sklearn.utils.validation import check_is_fitted
import copy
import multiprocessing as mp
import pandas as pd
import statistics
import itertools
import scipy.stats
from utils.modelling_methods.helper_functions import *
import pickle


def factor_quartile_norm_coef(X, cutoff_unique=5):
    norm_factor = dict()

    for col in X:
        if X[col].nunique() > cutoff_unique:
            if X.describe() is not None:
                norm_factor[col] = (X[col].describe().transpose()['75%'] - X[col].describe().transpose()['25%']) * 2
        else:
            norm_factor[col] = 1

    return norm_factor


def p_value_t_test_0(a):
    _, p_value = scipy.stats.ttest_1samp(a, popmean=0)
    return p_value


class SklearnBinaryClassification:

    def __init__(self, estimator):
        self.estimator = estimator
        self.params = self.estimator.get_params()
        self.log = dict()

        self.model_performance_train = None
        self.model_performance_test = None
        self.coef_dict = None
        self.coef_norm_dict = None

    def fit(self, X, y, X_test=None, y_test=None):

        self.estimator.fit(X, y)

        # Performance in training
        y_pred = self.estimator.predict(X)
        y_pred_prob = self.estimator.predict_proba(X)[:, 1]

        self.model_performance_train = sklearn_classification_metrics(y, y_pred, y_pred_prob)

        # Performance in test
        if (X_test is not None) & (y_test is not None):
            y_pred_test = self.estimator.predict(X_test)
            y_pred_prob_test = self.estimator.predict_proba(X_test)[:, 1]
            self.model_performance_test = sklearn_classification_metrics(y_test, y_pred_test, y_pred_prob_test)

        # Record co-efficients
        self.coef_dict = dict()
        self.coef_dict['Intercept'] = self.estimator.intercept_.item()
        self.coef_norm_dict = dict()
        self.coef_dict['Intercept'] = self.estimator.intercept_.item()

        coef_norm_factor = factor_quartile_norm_coef(X)

        for i, key in enumerate(X.columns):
            self.coef_dict[key] = self.estimator.coef_[0, i]
            self.coef_norm_dict['%s_norm' % key] = self.estimator.coef_[0, i] * coef_norm_factor[key]

        # Logging results
        self.log.update(self.model_performance_train)
        if (X_test is not None) & (y_test is not None):
            self.log.update(self.model_performance_test)
        self.log.update(self.coef_dict)
        self.log.update(self.coef_norm_dict)

    def get_params(self):
        return self.estimator.get_params()

    def set_params(self, **kwargs):
        # Check if params are allowed
        unknown_param = set(kwargs.keys()).difference(set(self.params.keys()))
        if len(unknown_param) > 0:
            raise KeyError('Parameter not existed: %s' % unknown_param)

        # Set the new params
        self.params.update(kwargs)
        self.estimator.set_params(**self.params)

    def get_estimator(self):
        check_is_fitted(self.estimator)
        return self.estimator

    def save_estimator(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_estimator(), f)


class StatsLRModel:

    def __init__(self):
        self.params = {
            'maxiter': 1000,
            'method': 'bfgs',
            'alpha': 1,
            'trim_mode': 'off'
        }

        self.log = dict()
        self.log.update(self.params)

        self.model = None
        self.model_metrics = None
        self.coef_summary = None
        self.coef_sig = None

    def fit(self, X, y):
        data = X.copy()
        data[y.name] = y.copy()

        formula = '%s ~ %s' % (y.name, ' + '.join(X.columns))

        self.model = logit(formula, data).fit(maxiter=self.params['maxiter'], method=self.params['method'])

        self.model_metrics = statsmodels_model_metrics(self.model)
        self.coef_summary = self.model.summary2().tables[1]
        self.coef_sig = self.coef_summary.to_dict()['Coef.']

        self.log.update(self.model_metrics)
        self.log.update(self.coef_sig)

    def fit_regularized(self, X, y):
        data = X.copy()
        data[y.name] = y.copy()

        formula = '%s ~ %s' % (y.name, ' + '.join(X.columns))

        self.model = logit(formula, data).fit_regularized(**self.params)

        self.model_metrics = statsmodels_model_metrics(self.model)
        self.coef_summary = self.model.summary2().tables[1]
        self.coef_sig = self.coef_summary.to_dict()['Coef.']

        self.log.update(self.model_metrics)
        self.log.update(self.coef_sig)

    def get_params(self):
        return self.params

    def set_params(self, **kwargs):
        # Remove unknown params
        common_params = list(set(kwargs.keys()).intersection(set(self.params.keys())))
        if len(common_params) > 0:
            for key in common_params:
                self.params[key] = kwargs.get(key)

        # logging the change
        self.log.update(self.params)

    def get_univarite_or_and_ci(self):
        metrics = {
            'OR': 2.718281828459045 ** self.model.summary2().tables[1].iloc[1, 0],
            'CI_95_lower': 2.718281828459045 ** self.model.summary2().tables[1].iloc[1, 4],
            'CI_95_upper': 2.718281828459045 ** self.model.summary2().tables[1].iloc[1, 5]
        }
        return metrics


class LRCrossValidation:

    repeat_counter = mp.Value('i', 0)

    def __init__(self, X, y, splitter=RepeatedKFold(), n_jobs=None):
        self.X = X
        self.y = y
        self.n_jobs = n_jobs

        self.model = SklearnBinaryClassification(estimator=LogisticRegression())
        self.params = self.model.estimator.get_params()

        self.log = dict()
        self.log.update(self.params)
        self.log.update(
            {
                'cv_total': len(y),
                'cv_P': sum(y),
                'cv_N': sum(1 - y)
            })

        self.splitter = splitter
        self.cv_map = list(self.splitter.split(self.X, self.y))
        self.n_repeats = len(self.cv_map)

        self.cv_results_full = None
        self.cv_summary_mean = None
        self.cv_summary_stdev = None
        self.cv_summary_CI_lower = None
        self.cv_summary_CI_upper = None
        self.cv_summary_sig = None
        self.cv_summary_sig_corrected = None

    def get_params(self):
        return self.params

    def set_params(self, **kwargs):
        # Check if params are allowed
        unknown_param = set(kwargs.keys()).difference(set(self.params.keys()))
        if len(unknown_param) > 0:
            raise KeyError('Parameter not existed: %s' % unknown_param)

        # Set the new params
        self.params.update(kwargs)

        # logging the change
        self.log.update(self.params)

    def _cv_core(self, train_index, test_index):

        X_train = self.X.iloc[train_index, :]
        y_train = self.y.iloc[train_index]
        X_test = self.X.iloc[test_index, :]
        y_test = self.y.iloc[test_index]

        if (len(y_train.unique()) > 1) & (len(y_test.unique()) > 1):
            cv_batch_metrics = dict()
            cv_batch_metrics['repeat_index'] = self.repeat_counter.value

            model = copy.deepcopy(self.model)
            model.set_params(**self.params)
            model.fit(X_train, y_train, X_test, y_test)

            cv_batch_metrics.update(model.log)

            with self.repeat_counter.get_lock():
                self.repeat_counter.value += 1
                print('Current repeat: (%s/%s)' % (self.repeat_counter.value, self.n_repeats), end='\r')

            return cv_batch_metrics
        else:
            with self.repeat_counter.get_lock():
                self.repeat_counter.value += 1
                print('Current repeat: (%s/%s) Single class outcome resulted from cv!'
                      % (self.repeat_counter.value, self.n_repeats), end='\r')
            return None

    def cross_validation(self):
        self.repeat_counter.value = 0

        if self.n_jobs == 1:
            cv_results_list = [self._cv_core(*index) for index in self.cv_map]
        else:
            pool = mp.Pool(self.n_jobs)
            cv_results_list = pool.starmap(self._cv_core, self.cv_map)
            pool.close()

        cv_results_list = [i for i in cv_results_list if i is not None]
        self.cv_results_full = pd.DataFrame.from_records(cv_results_list, index='repeat_index')

        self.cv_summary_mean = self.cv_results_full.apply(statistics.mean).to_dict()
        self.cv_summary_stdev = {'%s_stdev' % key: value
                                 for key, value in self.cv_results_full.apply(statistics.stdev).to_dict().items()}
        self.cv_summary_CI_lower = {'%s_CI95_lower' % key: value
                                    for key, value in self.cv_results_full.apply(CI95_lower).to_dict().items()}
        self.cv_summary_CI_upper = {'%s_CI95_upper' % key: value
                                    for key, value in self.cv_results_full.apply(CI95_upper).to_dict().items()}

        # sig_correction = 1

        # sig_correction *= len(self.X.columns)

        self.cv_summary_sig = {'%s_p_x!=0' % key: value
                               for key, value in
                               self.cv_results_full.apply(p_value_t_test_0).to_dict().items()
                               if key in list(self.X.columns)}
        # self.cv_summary_sig_corrected = {'%s_p_x!=0_cr' % key: value * sig_correction
        #                                  for key, value in self.cv_results_full.apply(p_value_t_test_0).to_dict().items()
        #                                  if key in list(self.X.columns) + ['Intercept']}

        self.log.update(self.cv_summary_mean)
        self.log.update(self.cv_summary_stdev)
        self.log.update(self.cv_summary_CI_lower)
        self.log.update(self.cv_summary_CI_upper)
        self.log.update(self.cv_summary_sig)
        # self.log.update(self.cv_summary_sig_corrected)

        print('')


class GridTuning:

    @staticmethod
    def expand_grid(params):
        """
        :param params: dict of range of paramsrw_value_df
        :return: df with all combinations, column names being names of params
        """
        for key in params.keys():
            if not isinstance(params[key], list):
                params[key] = [params[key]]
        iter_dict = pd.DataFrame.from_records(itertools.product(*params.values()),
                                              columns=params.keys()).to_dict('records')
        return iter_dict

    def __init__(self, cv, params_range):
        self.cv = cv
        self.params = self.cv.get_params()
        self.params.update(params_range)
        self.tuning_keys = list(params_range.keys())
        self.tune_params = self.expand_grid(self.params)
        self.tune_results = None
        self.best_params = None
        self.best_model = None

    def _tune_core(self, params):
        tune_metrics = dict()
        tune_metrics.update(params)

        msg = 'Current params: '
        for key in self.tuning_keys:
            msg += ' %s=%s ' % (key, params[key])

        print(msg)

        cv = copy.deepcopy(self.cv)
        cv.set_params(**params)
        cv.cross_validation()

        tune_metrics.update(cv.log)
        return tune_metrics

    def tuning(self):
        tune_result_list = [self._tune_core(params) for params in self.tune_params]
        self.tune_results = pd.DataFrame.from_records(tune_result_list)

    def get_best_performance(self, metrics='roc_auc_score'):
        best_model_index = self.tune_results.index[self.tune_results[metrics].rank(
                            method='min', ascending=False) == 1].values[0]
        return self.tune_results.iloc[best_model_index, :].to_dict()

    def get_feature_selection_dict(self, metrics='roc_auc_score', must_in=None):
        metrics = self.get_best_performance(metrics=metrics)
        feature_selection_dict = {key.replace('_p_x!=0', ''): (value < 0.05) & (value is not None)
                                  for key, value in metrics.items() if '_p_x!=0' in key}
        if must_in is not None:
            for feature in must_in:
                feature_selection_dict[feature] = True

        return feature_selection_dict

    def get_feature_selection(self, metrics='roc_auc_score', must_in=None):
        feature_selection_dict = self.get_feature_selection_dict(metrics=metrics, must_in=must_in)
        feature_list = [key for key, value in feature_selection_dict.items() if value]
        return feature_list

    def get_best_params(self, metrics='roc_auc_score'):
        best_model_index = self.tune_results.index[self.tune_results[metrics].rank(
                            method='min', ascending=False) == 1].values[0]
        self.best_params = self.tune_params[best_model_index]

        return self.best_params

    def get_best_model(self, metrics='roc_auc_score'):
        self.best_params = self.get_best_params(metrics)

        best_model = copy.deepcopy(self.cv.model)
        best_model.set_params(**self.best_params)

        best_model.fit(self.cv.X, self.cv.y)
        self.best_model = best_model

        return self.best_model

    def get_best_estimator(self, metrics='roc_auc_score'):
        best_model = self.get_best_model(metrics=metrics)
        return best_model.estimator


# from utils.IO import DataByContext, DataInput

# data_input = DataInput()
# data_input.filter(inclusion_filter='not_readmission')
# data = DataByContext(feature_choice='basic_condi_symp', outcome_choice='poor_prognosis', data_input=data_input)
#
# X = data.X
# y = data.y

# statsmodel = StatsLRModel()
# statsmodel.set_params(alpha=1, method='l1')
# statsmodel.fit_regularized(X, y)
# statsmodel.coef_summary

#
# cv = LRCrossValidation(data.X, data.y)
# cv.set_params(max_iter=5000, penalty='l1', solver='liblinear')

# cv.cross_validation()
# print(cv.log)

# resample(data.X, n_samples=2)

# params_range_l1 = {
#    'C': [2**i for i in range(-2, 2)],
#    'max_iter': 5000,
#    'penalty': 'l1',
#    'solver': 'liblinear'
# }

# cv = LRCrossValidation(data.X, data.y, splitter=)
# tune = GridTuning(cv=cv, params_range=params_range_l1)

# tune.tuning()

# tune.get_best_performance()
# tune.get_best_params()

# tune.get_best_model()
