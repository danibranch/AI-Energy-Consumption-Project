from datetime import datetime

import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler


def parse_timestamp(time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))


def prepare_data(dataset, save_to_csv=False):
    dataset.columns = ['consumption', 'coal', 'gas', 'hidroelectric',
                       'nuclear', 'wind', 'solar', 'biomass', 'productions']
    dataset.index.name = 'date'
    dataset = dataset[555:]
    if save_to_csv:
        dataset.to_csv('dataset/prepared_train_electricity.csv')
    return dataset


def scale_data(dataset, save_to_csv=False):
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit(dataset)
    scaler.fit_transform(dataset)
    if save_to_csv:
        dataset.to_csv('dataset/scaled_train_electricity.csv')
    return dataset


def create_plots():
    dataset = pd.read_csv(
        'dataset/prepared_train_electricity.csv', header=0, index_col=0)
    values = dataset.values
    groups = [1, 2, 3, 4, 5, 6, 7]  # coal to biomass
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()


def create_consumption_production_plot():
    dataset = pd.read_csv(
        'dataset/prepared_train_electricity.csv', header=0, index_col=0)
    dataset = dataset.head(5)
    values = dataset.values
    groups = [0, 8]  # coal to productions

    pyplot.figure()
    for group in groups:
        pyplot.plot(dataset.index[:], values[:, group])

    pyplot.show()


def rearrange_cols(dataset, save_to_csv=False):
    cols = dataset.columns.tolist()
    cols = cols[1:] + cols[0:1]
    dataset = dataset[cols]
    if save_to_csv:
        dataset.to_csv('dataset/prepared_train_electricity_fin2.csv')
    return dataset


def remove_outliers(dataset, save_to_csv=False):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) |
                        (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    if save_to_csv:
        dataset.to_csv('dataset/prepared_train_electricity.csv')
    return dataset

# scale_data()
# create_plots()
