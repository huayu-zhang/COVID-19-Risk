import pandas as pd
import scipy.stats
import numpy as np
from utils.vis.conti_cat import boxplot, violinplot
from utils.modelling_methods.logistic_regression import StatsLRModel


class CrossTabAnalysis:

    def __init__(self, condition, case):
        self.condition = condition
        self.case = case

        self._cross_table = pd.crosstab(condition, case)
        self._cross_table_array = self._cross_table.to_numpy()

        self._cross_table_index_norm = pd.crosstab(condition, case, normalize='index')
        self._cross_table_index_norm_array = self._cross_table_index_norm.to_numpy()

        self._chi2, self._p_value, _, _ = scipy.stats.chi2_contingency(self._cross_table_array)
        self._risk_ratio = self._cross_table_index_norm_array[1, 1]/self._cross_table_index_norm_array[0, 1]
        self._odds_ratio = self._cross_table_array[1, 1] * self._cross_table_array[0, 0] / (
                self._cross_table_array[1, 0] * self._cross_table_array[0, 1])

        self._log_OR = np.log(self._odds_ratio)
        self._log_OR_SE = np.sqrt(np.sum(1/self._cross_table_array))

        self._log_OR_CI95 = np.array([self._log_OR - self._log_OR_SE * 1.96, self._log_OR + self._log_OR_SE * 1.96])
        self._OR_CI95 = np.exp(self._log_OR_CI95)

    def get_cross_table(self, margins=False, normalize=False):
        return pd.crosstab(self.condition, self.case, margins=margins, normalize=normalize)

    def get_chi2(self):
        return self._chi2

    def get_p_value(self):
        return self._p_value

    def get_odds_ratio(self):
        return self._odds_ratio

    def get_risk_ratio(self):
        return self._risk_ratio

    def get_OR_CI95(self):
        return self._OR_CI95

    def get_summary(self):

        summary = {
            'case': self.case.name,
            'condition': self.condition.name,
            'total_n': len(self.case),
            'N': sum(1 - self.case),
            'P': sum(self.case),
            'non_non': self._cross_table_array[0, 0],
            'case_non': self._cross_table_array[0, 1],
            'non_condition': self._cross_table_array[1, 0],
            'case_condition': self._cross_table_array[1, 1],
            'risk_ratio': self._risk_ratio,
            'odds_ratio': self._odds_ratio,
            'OR_CI95_lower': self._OR_CI95[0],
            'OR_CI95_upper': self._OR_CI95[1],
            'chi2': self._chi2,
            'p_value': self._p_value
        }

        return summary


class BinaryCatOnConti:

    def __init__(self, cat_outcome, conti_feature, extra_cat=None):

        self.outcome = cat_outcome
        self.feature = conti_feature
        self.extra_cat = extra_cat

        feature_groups = {
            0: self.feature[self.outcome == 0],
            1: self.feature[self.outcome == 1]
        }

        self.summary = dict()
        self.summary['outcome'] = self.outcome.name
        self.summary['feature'] = self.feature.name

        self.summary['outcome_freq_0'] = sum(1 - self.outcome)
        self.summary['outcome_freq_1'] = sum(self.outcome)

        self.summary['mean_0'] = np.mean(feature_groups[0])
        self.summary['mean_1'] = np.mean(feature_groups[1])
        self.summary['ratio'] = self.summary['mean_1'] / self.summary['mean_0']

        self.summary['median_0'] = np.median(feature_groups[0])
        self.summary['Q1_0'] = np.quantile(feature_groups[0], 0.25)
        self.summary['Q3_0'] = np.quantile(feature_groups[0], 0.75)

        self.summary['median_1'] = np.median(feature_groups[1])
        self.summary['Q1_1'] = np.quantile(feature_groups[1], 0.25)
        self.summary['Q3_1'] = np.quantile(feature_groups[1], 0.75)

        lr = StatsLRModel()
        lr.fit(pd.DataFrame(conti_feature), cat_outcome)
        lr_metrics = lr.get_univarite_or_and_ci()

        self.summary['OR'] = lr_metrics['OR']
        self.summary['CI_95_lower'] = lr_metrics['CI_95_lower']
        self.summary['CI_95_upper'] = lr_metrics['CI_95_upper']

        self.summary['stdev_0'] = np.std(feature_groups[0])
        self.summary['stdev_1'] = np.std(feature_groups[1])

        self.summary['u_test'] = self.feature.nunique() <= 6

        if not self.summary['u_test']:

            _, self.summary['p_equal_var_levene'] = scipy.stats.levene(feature_groups[0],
                                                                       feature_groups[1])

            if self.summary['p_equal_var_levene'] > 0.05:
                _, self.summary['p_value'] = scipy.stats.ttest_ind(feature_groups[0], feature_groups[1],
                                                                   equal_var=True)
            else:
                _, self.summary['p_value'] = scipy.stats.ttest_ind(feature_groups[0], feature_groups[1],
                                                                   equal_var=False)
        else:
            self.summary['p_equal_var_levene'] = pd.NaT
            _, self.summary['p_value'] = scipy.stats.mannwhitneyu(feature_groups[0], feature_groups[1])

    def get_summary(self):
        return self.summary

    def get_boxplot(self, h_line=None, add_hue=False):
        if add_hue:
            hue_series = self.extra_cat
        else:
            hue_series = None

        fig = boxplot(category_series=self.outcome,
                      continuous_series=self.feature,
                      h_line=h_line,
                      hue_series=hue_series)
        return fig

    def get_violinplot(self, h_line=None, add_hue=False):
        if add_hue:
            hue_series = self.extra_cat
        else:
            hue_series = None

        fig = violinplot(category_series=self.outcome,
                         continuous_series=self.feature,
                         h_line=h_line,
                         hue_series=hue_series)
        return fig


# from utils.IO import DataInput
# from utils.IO import DataInput
# from statsmodels.api import Logit

# data_input = DataInput()
# data = data_input.select_columns(['age', 'death'])

# cc = BinaryCatOnConti(conti_feature=data['age'], cat_outcome=data['ards_severe'], extra_cat=data['male'])
#
# cc.get_boxplot(add_hue=True).show()
