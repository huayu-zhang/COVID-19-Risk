from utils.modelling_methods.helper_functions import sklearn_classification_metrics
import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import RepeatedKFold
import copy


def index_of_value_closest(a, value):
    a = np.array(a)
    return np.argmin(np.abs(a - value))


class ProbLiftingMetrics:

    def __init__(self, estimator, X, y, X_test=None, y_test=None):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test

        if self.X_test is None:
            self.X_test = self.X
        if self.y_test is None:
            self.y_test = self.y

        self.y_pred_prob = self.estimator.predict_proba(self.X_test)[:, 1]

        self.results_df = None

    def prob_lifting(self, step=0.01):

        prob_cutoffs = [step * i for i in range(0, int(1/step) + 1)]

        results = list()

        for cutoff in prob_cutoffs:
            log = {'prob_cutoff': cutoff}
            y_pred_cutoff = (self.y_pred_prob > cutoff).astype('int')
            log.update(sklearn_classification_metrics(y_true=self.y_test,
                                                      y_pred=y_pred_cutoff,
                                                      y_pred_prob=self.y_pred_prob))
            results.append(log)

        self.results_df = pd.DataFrame.from_records(results)

    def get_results(self):
        return self.results_df

    def get_target_prob(self, target, target_value):
        min_index = np.argmin(np.abs(self.results_df[target] - target_value))
        target_prob = self.results_df['prob_cutoff'].iloc[min_index]
        return target_prob


class ProbLiftingCV:

    repeat_counter = mp.Value('i', 0)

    def __init__(self, X, y, estimator, splitter=RepeatedKFold(), step=0.001, n_jobs=None):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.step = step
        self.params = self.estimator.get_params()

        self.splitter = splitter
        self.cv_map = list(self.splitter.split(self.X, self.y))
        self.n_repeats = len(self.cv_map)

        self.cv_summary = None

    def get_params(self):
        return self.params

    def set_params(self, **kwargs):
        # Check if params are allowed
        unknown_param = set(kwargs.keys()).difference(set(self.params.keys()))
        if len(unknown_param) > 0:
            raise KeyError('Parameter not existed: %s' % unknown_param)

        # Set the new params
        self.params.update(kwargs)

    def _cv_core(self, train_index, test_index):

        X_train = self.X.iloc[train_index, :]
        y_train = self.y.iloc[train_index]
        X_test = self.X.iloc[test_index, :]
        y_test = self.y.iloc[test_index]

        if (len(y_train.unique()) > 1) & (len(y_test.unique()) > 1):

            estimator = copy.deepcopy(self.estimator)
            estimator.set_params(**self.params)
            estimator.fit(X_train, y_train)

            pl_method = ProbLiftingMetrics(estimator=estimator, X=X_train, y=y_train,
                                           X_test=X_test, y_test=y_test)
            pl_method.prob_lifting(step=self.step)

            with self.repeat_counter.get_lock():
                self.repeat_counter.value += 1
                print('Current repeat: (%s/%s)' % (self.repeat_counter.value, self.n_repeats), end='\r')
            return pl_method.get_results()
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
        cv_summary = sum(cv_results_list)/len(cv_results_list)

        self.cv_summary = cv_summary

        print('')

    def get_summary(self):
        return self.cv_summary


# from utils.IO import DataByContext
# from sklearn.linear_model import LogisticRegression


# data = DataByContext(feature_choice='basic_lab', outcome_choice='death')

# estimator = LogisticRegression(max_iter=5000, penalty='l1', C=16, solver='liblinear')
# estimator.fit(data.X, data.y)

# pl = ProbLiftingMetrics(estimator=estimator, X=data.X, y=data.y)
# pl.prob_lifting()
# pl.results_df.to_csv('aaa.csv')
# pl.get_target_prob(target='sensitivity', target_value=0.7)

# pl_cv = ProbLiftingCV(X=data.X, y=data.y, estimator=estimator)
# pl_cv.set_params(max_iter=5000)
# pl_cv.cross_validation()
# pl_cv.get_summary().to_csv('aaa.csv')
