import sys
import os
sys.path.extend(os.getcwd())
import pickle
from src.risk_stratification import RiskStratificationBinaryOutcome
import pandas as pd
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Risk Stratification')
    parser.add_argument('estimator', type=str,
                        help='file name of estimator to load')
    parser.add_argument('X', default='X.csv',
                        help='Dataframe of features')
    parser.add_argument('y', default='y.csv',
                        help='Dataframe of outcome')
    parser.add_argument('cutoff_low', type=float,
                        help='lower cutoff between low and mid')
    parser.add_argument('cutoff_high', type=float,
                        help='higher cutoff between mid and high')
    parser.add_argument('outcome_0', type=str,
                        help='y label on left figure')
    parser.add_argument('outcome_1', type=str,
                        help='y label on right figure')
    parser.add_argument('--x_label', type=str, default='Predicted Probability\nGroup DL',
                        help='x label')
    parser.add_argument('--figure_title', type=str, default='Risk Stratification in Validation Cohort')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.estimator, 'rb') as f:
        estimator = pickle.load(f)

    X = pd.read_csv(args.X, index_col=0)
    y = pd.read_csv(args.y, index_col=0).iloc[:,0]

    log = {
        'estimator': args.estimator
    }

    name_0 = args.outcome_0.replace('_', ' ')
    name_1 = args.outcome_1.replace('_', ' ')

    rs = RiskStratificationBinaryOutcome(fitted_estimator=estimator,
                                         X=X,
                                         y=y,
                                         outcome_names=[name_0, name_1])

    rs.set_LMH_bins(LMH_bins=[0, args.cutoff_low, args.cutoff_high, 1])

    rs.get_visualization(savefig_path='%s_rs_histogram_%s_%s.svg' % (args.estimator, args.cutoff_low, args.cutoff_high),
                         x_label='Predicted probability\nGroup DL', title=args.figure_title)
    log.update(rs.get_summary())

    pd.DataFrame.from_records([log]).to_csv('%s_results_%s_%s.o' % (args.estimator, args.cutoff_low, args.cutoff_high))


if __name__ == '__main__':
    main()
