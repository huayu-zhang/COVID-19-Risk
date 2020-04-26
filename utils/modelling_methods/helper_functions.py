import sklearn.metrics
import numpy as np
import scipy.stats


def sklearn_classification_metrics(y_true, y_pred, y_pred_prob):

    metrics_dict = dict()

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    metrics_dict['total'] = len(y_true)

    metrics_dict['P'] = sum(y_true)
    metrics_dict['N'] = sum(1 - y_true)

    metrics_dict['TN'], metrics_dict['FP'], metrics_dict['FN'], metrics_dict['TP'] = confusion_matrix.ravel()

    metrics_dict['sensitivity'] = metrics_dict['TP']/metrics_dict['P']
    metrics_dict['specificity'] = metrics_dict['TN']/metrics_dict['N']

    if metrics_dict['TP'] + metrics_dict['FP'] == 0:
        metrics_dict['PPV'] = 0
    else:
        metrics_dict['PPV'] = metrics_dict['TP']/(metrics_dict['TP'] + metrics_dict['FP'])

    if metrics_dict['TN'] + metrics_dict['FN'] == 0:
        metrics_dict['NPV'] = 0
    else:
        metrics_dict['NPV'] = metrics_dict['TN']/(metrics_dict['TN'] + metrics_dict['FN'])

    metrics_dict['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    metrics_dict['f1_score'] = sklearn.metrics.f1_score(y_true, y_pred)

    metrics_dict['roc_auc_score'] = sklearn.metrics.roc_auc_score(y_true, y_pred_prob)

    return metrics_dict


def statsmodels_model_metrics(stats_model):

    selected_metrics = ['prsquared', 'llr_pvalue', 'llnull', 'llf']
    key_names = ['pseudo_r2', 'p_value', 'log-likelihood_null', 'log-likelihood']
    metrics_dict = {name: getattr(stats_model, key) for name, key in zip(key_names, selected_metrics)}
    metrics_dict['log-likelihood-diff'] = metrics_dict['log-likelihood'] - metrics_dict['log-likelihood_null']

    return metrics_dict


def confidence_interval(a, confidence=0.95, lower=True, upper=True):
    a = 1.0 * np.array(a)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    if lower:
        if upper:
            return m-h, m+h
        else:
            return m-h
    else:
        if upper:
            return m+h
        else:
            raise ValueError('At least choose to return one bound: lower or upper')


def CI95_lower(a):
    return confidence_interval(a, lower=True, upper=False)


def CI95_upper(a):
    return confidence_interval(a, lower=False, upper=True)

