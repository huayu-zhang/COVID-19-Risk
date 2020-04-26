import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def boxplot(category_series, continuous_series, h_line=None, hue_series=None):

    data = pd.concat([category_series, continuous_series], axis=1)
    category_name = category_series.name
    continuous_name = continuous_series.name
    hue_name = None

    if hue_series is not None:
        data = pd.concat([data, hue_series], axis=1)
        hue_name = hue_series.name

    fig, ax = plt.subplots()
    sns.boxplot(x=category_name, y=continuous_name, hue=hue_name, data=data, ax=ax)

    if h_line is not None:
        ax.axhline(y=h_line, ls='--')
    plt.close()

    return fig


def violinplot(category_series, continuous_series, h_line=None, hue_series=None):

    data = pd.concat([category_series, continuous_series], axis=1)
    category_name = category_series.name
    continuous_name = continuous_series.name

    hue_name = None

    if hue_series is not None:
        data = pd.concat([data, hue_series], axis=1)
        hue_name = hue_series.name

    fig, ax = plt.subplots()
    sns.violinplot(x=category_name, y=continuous_name, data=data, hue=hue_name, split=True, ax=ax)

    if h_line is not None:
        ax.axhline(y=h_line, ls='--', color='black')
    plt.close()

    return fig


# from utils.IO import DataInput
#
# data_input = DataInput()
# data = data_input.select_columns(['age', 'ards_severe', 'male'])
#
# fig = boxplot(category_series=data['ards_severe'], continuous_series=data['age'])
