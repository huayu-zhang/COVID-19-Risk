import sys
import os

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

import json
import pandas as pd
import time
from scipy.stats import zscore
from numpy import abs
from utils.feature_outcome_definition import AdmissionContext


with open('project_params.json', 'r') as f:
    params = json.load(f)


def get_params(params):
    return params


def save_params(params):
    os.system('cp project_params.json project_params_%s.json' % time.time())
    with open('project_params.json', 'w') as f:
        json.dump(params, f, indent=4)


class DataInput:

    def __init__(self, file='DATA', description='COLUMN_DESCRIPTION'):
        self.data = pd.read_csv(params[file], sep='\t', index_col=0)
        self.data = self.data[pd.to_datetime(self.data.date_admission) >= pd.to_datetime('2020-02-01')]
        self.column_description = pd.read_csv(params[description], sep='\t', index_col=0)

    def filter(self, inclusion_filter):
        self.data = self.data[self.data[inclusion_filter] == 1]

    def select_index(self, index, inplace=False):
        new_data = self.data.copy()
        new_data = new_data.loc[index, :]

        if inplace:
            self.data = new_data

        return new_data

    def select_columns(self, columns, inplace=False, complete_cases=True, remove_outliers=False):

        if isinstance(columns, str):
            columns = [columns]
        new_data = self.data.copy()
        new_data = new_data[columns]

        if complete_cases:
            new_data.dropna(inplace=True)

        if remove_outliers:
            continuous_columns = [col for col in new_data if new_data[col].nunique() > 5]
            not_outliers = (abs(zscore(new_data[continuous_columns])) < 5).all(axis=1)
            new_data = new_data[not_outliers]

        if inplace:
            self.data = new_data

        return new_data

    def index_with_complete_cases(self, columns):
        return self.select_columns(columns=columns).index

    def columns_by_type(self, type):
        return list(self.column_description.column_name[self.column_description.type == type])

    def columns_by_var_type(self, var_type):
        return list(self.column_description.column_name[self.column_description.var_type == var_type])


class DataByContext:

    def __init__(self, feature_choice, outcome_choice, data_input=DataInput(), context=AdmissionContext()):

        # Define features and outcomes
        self.feature_choice = feature_choice
        self.outcome_choice = outcome_choice

        self.feature = context.feature_dict[feature_choice]
        self.outcome = context.outcome_dict[outcome_choice]

        # Get subset
        self.data_subset = data_input.select_columns(self.feature + self.outcome)

        # sklearn model input
        self.X = self.data_subset[self.feature]
        self.y = self.data_subset[self.outcome[0]]

        # statsmodels model input
        self.formula = '%s ~ %s' % (self.outcome[0], ' + '.join(self.feature))
