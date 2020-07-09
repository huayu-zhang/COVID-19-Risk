from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


class RiskStratificationBinaryOutcome:

    def __init__(self, fitted_estimator, X, y, outcome_names=None, bin_width=20):

        if outcome_names is None:
            outcome_names = ['0', '1']

        check_is_fitted(fitted_estimator)
        self.estimator = fitted_estimator

        self.X = X
        self.y = y
        self.outcome_names = outcome_names
        self.bin_width = bin_width

        self.y_pred_prob = self.estimator.predict_proba(X)[:, 1]
        self.Y_true_prob = pd.DataFrame({'true': self.y, 'pred_prob': self.y_pred_prob})

        self.event_rate_df = event_rate_by_prob_cutoffs(y_true=self.y, y_pred_prob=self.y_pred_prob,
                                                        cutoffs=prob_cutoffs_by_bins(self.y_pred_prob, bin_width=bin_width))

        self.LMH_bins = None
        self._target_low_risk = None
        self._target_high_risk = None

    def stratify_by_target_rate(self, target_low_risk, target_high_risk):

        self.LMH_bins = bins_by_target_rate(
            y_true=self.y, y_pred_prob=self.y_pred_prob,
            rate_low_risk=target_low_risk, rate_high_risk=target_high_risk,
            bin_width=self.bin_width)

        self._target_low_risk = target_low_risk
        self._target_high_risk = target_high_risk

        return self.LMH_bins

    def set_LMH_bins(self, LMH_bins):
        self.LMH_bins = LMH_bins

    def get_summary(self):

        summary = {
            'outcome': self.outcome_names[1],
            'sample_rate': np.mean(self.y),
            'target_low_risk': self._target_low_risk,
            'target_high_risk': self._target_high_risk
        }

        for i, bin in enumerate(self.LMH_bins):
            summary['bin_%s' % i] = bin

        counts_in_3bins, _, _ = plt.hist(x=[self.Y_true_prob.pred_prob[self.Y_true_prob.true == 0],
                                            self.Y_true_prob.pred_prob[self.Y_true_prob.true == 1]],
                                         bins=self.LMH_bins, stacked='true')
        plt.close()

        rate_in_bins = (counts_in_3bins[1] - counts_in_3bins[0])/counts_in_3bins[1]
        groups = ['low_risk', 'mid_risk', 'high_risk']

        for i, group in enumerate(groups):
            summary['%s_count' % group] = counts_in_3bins[1][i]
            summary['%s_outcome_percentage' % group] = rate_in_bins[i]

        return summary

    def get_analysis_plot(self, savefig_path=None):

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        df = self.event_rate_df
        log = self.get_summary()

        sns.lineplot(x='prob_cutoffs', y='value', hue='variable', data=df.melt(id_vars=['prob_cutoffs']), ax=axes[0])
        axes[0].set_ylabel('%outcome or %coverage')
        axes[0].set_xlabel('Predicted probability')
        axes[0].vlines([log['bin_1'], log['bin_2']], [0, 0],
                       [df.loc[df.prob_cutoffs == log['bin_1'], 'outcome%_below_cutoff'].item(),
                        df.loc[df.prob_cutoffs == log['bin_2'], 'outcome%_above_cutoff'].item()], ls='--')
        axes[0].hlines([df.loc[df.prob_cutoffs == log['bin_1'], 'outcome%_below_cutoff'].item(),
                        df.loc[df.prob_cutoffs == log['bin_2'], 'outcome%_above_cutoff'].item()], [0, 0],
                       [log['bin_1'], log['bin_2']], ls='--')

        axes[1].hist(x=[self.Y_true_prob.pred_prob[self.Y_true_prob.true == 0],
                        self.Y_true_prob.pred_prob[self.Y_true_prob.true == 1]],
                     bins=self.LMH_bins, stacked='true')
        axes[0].set_ylabel('Counts')
        axes[0].set_xlabel('Predicted probability')

        if savefig_path is not None:
            fig.savefig(savefig_path)

        plt.close()

        return fig, axes

    def get_visualization(self, savefig_path=None,
                          title='Risk Stratification Derivation Cohort', x_label='Predicted probability'
                          ):

        counts_in_bins, _, _ = plt.hist(x=[self.Y_true_prob.pred_prob[self.Y_true_prob.true == 0],
                                           self.Y_true_prob.pred_prob[self.Y_true_prob.true == 1]],
                                        bins=self.event_rate_df.prob_cutoffs, stacked='true')

        plt.close()

        data_for_stack_area = pd.DataFrame({self.outcome_names[0]: counts_in_bins[0]/counts_in_bins[1],
                                            self.outcome_names[1]: 1 - counts_in_bins[0]/counts_in_bins[1]})

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

        axes[0].stackplot(self.event_rate_df.prob_cutoffs[1:],
                          data_for_stack_area[self.outcome_names[0]], data_for_stack_area[self.outcome_names[1]],
                          labels=[self.outcome_names[0], self.outcome_names[1]])

        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel('%s (%s)' % (self.outcome_names[1], '%'))
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles[::-1], labels[::-1], loc='lower left')

        axes[1].hist(x=[self.Y_true_prob.pred_prob[self.Y_true_prob.true == 0],
                        self.Y_true_prob.pred_prob[self.Y_true_prob.true == 1]],
                     bins=self.LMH_bins, stacked='true')

        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('Counts')

        fig.suptitle(title, fontsize=10, fontweight='bold', fontname='Times New Roman')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig_path is not None:
            fig.savefig(savefig_path)

        plt.close()

        return fig, axes


def prob_cutoffs_by_bins(y_pred_prob, bin_width=20, offset=0.0):

    sorted_probs = np.sort(np.array(y_pred_prob))

    total_number = len(sorted_probs)

    bin_cutoffs = [0.0]
    pointer = bin_width

    while pointer < total_number:
        bin_cutoffs.append(sorted_probs[pointer] + offset)
        pointer += bin_width

    bin_cutoffs.append(1.0)

    return np.array(bin_cutoffs)


def event_rate_by_prob_cutoffs(y_true, y_pred_prob, cutoffs):

    Y_true_prob = pd.DataFrame({'true': y_true, 'pred_prob': y_pred_prob})
    Y_true_prob.sort_values(by='pred_prob')

    n_sample = len(y_true)
    means_in_bins, _, _ = scipy.stats.binned_statistic(x=Y_true_prob.pred_prob,
                                                       values=Y_true_prob.true,
                                                       statistic='mean',
                                                       bins=cutoffs)
    counts_in_bins, _, _ = scipy.stats.binned_statistic(x=Y_true_prob.pred_prob,
                                                        values=Y_true_prob.true,
                                                        statistic='count',
                                                        bins=cutoffs)

    n_bins = len(means_in_bins)
    event_rate_below_cutoff = [0]
    event_rate_above_cutoff = list()
    sample_coverage = [0]

    for i in range(n_bins):
        event_rate_below_cutoff.append(np.mean(means_in_bins[:(i + 1)]))
        event_rate_above_cutoff.append(np.mean(means_in_bins[i:]))
        sample_coverage.append(np.sum(counts_in_bins[:i]) / n_sample)
    event_rate_above_cutoff.append(event_rate_above_cutoff[-1])

    event_rate_df = pd.DataFrame({'prob_cutoffs': cutoffs,
                                  'outcome%_below_cutoff': event_rate_below_cutoff,
                                  'outcome%_above_cutoff': event_rate_above_cutoff,
                                  'sample_coverage%': sample_coverage
                                  })

    return event_rate_df


def last_below_cutoff(x, cutoff):
    decision_array = np.sign(np.array(x) - cutoff)
    index = len(x) - 1

    while (decision_array[index] > 0) & (index > 0):
        index -= 1

    if index == 0:
        raise ValueError('No value below cutoff')

    return index


def first_above_cutoff(x, cutoff):
    decision_array = np.sign(np.array(x) - cutoff)

    index = 0

    while (decision_array[index] < 0) & (index < len(decision_array) - 1):
        index += 1

    if index == len(decision_array):
        raise ValueError('No value above cutoff')

    return index


def bins_by_target_rate(y_true, y_pred_prob, rate_low_risk, rate_high_risk,
                        bin_width=10, offset=0, return_df=False):
    cutoffs = prob_cutoffs_by_bins(y_pred_prob=y_pred_prob, bin_width=bin_width, offset=offset)

    event_rate_df = event_rate_by_prob_cutoffs(y_true=y_true, y_pred_prob=y_pred_prob, cutoffs=cutoffs)

    cutoff_1 = event_rate_df.prob_cutoffs[last_below_cutoff(event_rate_df['outcome%_below_cutoff'],
                                                            cutoff=rate_low_risk)] + offset
    cutoff_2 = event_rate_df.prob_cutoffs[first_above_cutoff(event_rate_df['outcome%_above_cutoff'],
                                                             cutoff=rate_high_risk)] + offset

    if return_df:
        return [0, cutoff_1, cutoff_2, 1], event_rate_df
    else:
        return [0, cutoff_1, cutoff_2, 1]


# import pickle

# with open('poor_outcome_prediction/dump/' + 'tuning_experiments_l1_0410', 'rb') as f:
#     experiments = pickle.load(f)

# experiments = experiments['death:basic_lab']
#
# estimator = experiments.get_best_estimator()
#
#
# rs = RiskStratificationBinaryOutcome(fitted_estimator=estimator,
#                                      X=experiments.cv.X, y=experiments.cv.y,
#                                      outcome_names=['Survived', 'Death'])
#
# rs.stratify_by_target_rate(rate_low_risk=0.004, rate_high_risk=0.4)
#
# rs.get_summary()
#
# fig, axes = rs.get_visualization(savefig_path='test.svg')
# fig, axes = rs.get_analysis_plot()
# fig.show()
