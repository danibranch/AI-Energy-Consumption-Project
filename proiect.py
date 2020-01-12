from datetime import datetime

import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler


def parse_timestamp(time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))


def prepare_data(save_to_csv=False):
    dataset = pd.read_csv("dataset/train_electricity.csv",
                          parse_dates=['Date'], date_parser=parse_timestamp, index_col=0)

    dataset.columns = ['consumption', 'coal', 'gas', 'hidroelectric',
                       'nuclear', 'wind', 'solar', 'biomass', 'productions']
    dataset.index.name = 'date'
    dataset = dataset[555:]
    print(dataset.head(5))
    if save_to_csv:
        dataset.to_csv('dataset/prepared_train_electricity.csv')
    return dataset


def scale_data(save_to_csv=False):
    dataset = pd.read_csv(
        'dataset/prepared_train_electricity.csv', header=0, index_col=0)
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


def remove_outliers(save_to_csv=False):
    dataset = pd.read_csv(
        'dataset/prepared_train_electricity.csv', header=0, index_col=0)
    # iqr = stats.iqr(dataset, axis=0)

    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

    dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) |
                        (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]

    print(dataset)
    if save_to_csv:
        dataset.to_csv('dataset/prepared_train_electricity.csv')
    return dataset


# scale_data()
create_plots()
