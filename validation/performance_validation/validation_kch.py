import sklearn.metrics
import pickle
import argparse
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


def parse_args():

    parser = argparse.ArgumentParser(description='External_validation')
    parser.add_argument('estimator', type=str,
                        help='file name of estimator to load')
    parser.add_argument('X',
                        help='Dataframe of features')
    parser.add_argument('y',
                        help='Dataframe of outcome')
    parser.add_argument('--describe', default=False, type=bool,
                        help='Whether to return description of input')
    parser.add_argument('--return_y_prob', default=False, type=bool,
                        help='Whether to return true y and predicted prob')
    parser.add_argument('--prob_lifting', default=False, type=bool,
                        help='Whether to return prob_lifting results')
    parser.add_argument('--do_calibration', default=False, type=bool,
                        help='Whether to return calibration results')

    return parser.parse_args()


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


def prob_lifting(y_true, y_pred_prob, step=0.001):

    prob_cutoffs = [round(step * i, 3) for i in range(0, int(1/step) + 1)]

    results = list()

    for cutoff in prob_cutoffs:
        log = {'prob_cutoff': cutoff}
        y_pred_cutoff = (y_pred_prob > cutoff).astype('int')
        log.update(sklearn_classification_metrics(y_true=y_true,
                                                  y_pred=y_pred_cutoff,
                                                  y_pred_prob=y_pred_prob))
        results.append(log)

    results_df = pd.DataFrame.from_records(results)
    return results_df


def calibration_diagram(probs, y, model):
    # reliability diagram
    fop, mpv = calibration_curve(y, probs, n_bins=10, normalize=True)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    pyplot.plot(mpv, fop, marker='.')
    # pyplot.show()
    pyplot.savefig('./calibration_diag.png')


def do_calibration(X, y, model):
    # split into train/test sets
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

    # calibrate model on validation data
    calibrator = CalibratedClassifierCV(model, cv='prefit')
    calibrator.fit(trainX, trainy)
    # predict probabilities
    probs = calibrator.predict_proba(testX)[:, 1]
    # evaluate the model
    calibration_diagram(probs, testy, calibrator)
    return calibrator, testX, testy


def run_validation(args, X, Y, estimator):
    y_true = Y.iloc[:, 0]

    y_pred = estimator.predict(X)
    y_pred_prob = estimator.predict_proba(X)[:, 1]
    if not args.do_calibration:
        calibration_diagram(y_pred_prob, y_true, None)

    metrics = sklearn_classification_metrics(y_true=y_true, y_pred=y_pred, y_pred_prob=y_pred_prob)

    metrics_df = pd.DataFrame.from_records([metrics])
    metrics_df.to_csv('%s_validation_metrics.o' % args.estimator)

    # coefs = {'intercept': estimator.intercept_.item()}
    # for i, key in enumerate(X.columns):
    #     coefs[key] = estimator.coef_[0, i]

    # pd.DataFrame.from_records([coefs]).to_csv('coefs_csv')

    if args.describe:
        X.describe(include='all').transpose().to_csv('X_describe.o')
        Y.describe(include='all').transpose().to_csv('y_describe.o')

    if args.return_y_prob:
        y_true_y_prob = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
        y_true_y_prob.to_csv('y_true_y_prob.o')

    if args.prob_lifting:
        prob_lifting_df = prob_lifting(y_true, y_pred_prob)
        prob_lifting_df.to_csv('prob_lifting.o')


def main():
    args = parse_args()

    with open(args.estimator, 'rb') as f:
        estimator = pickle.load(f)

    X = pd.read_csv(args.X, index_col=0)
    Y = pd.read_csv(args.y, index_col=0)

    if args.do_calibration:
        estimator, X, Y = do_calibration(X, Y, estimator)

    run_validation(args, X, Y, estimator)


if __name__ == '__main__':
    main()
