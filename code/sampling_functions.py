import numpy as np
import pandas as pd
import logging


def run_sampling(stock, population, type, n_samples):
    if type == 'sequential':
        overlap_array = create_overlap_array(population)
        sample = sequential_sampling(population, overlap_array, n_samples)
    elif type == 'weighted_random':
        overlap_array = create_overlap_array(population)
        weights = create_uniqueness_variable(overlap_array)
        sample = weighted_random_sampling(population, n_samples, weights)
    elif type == 'random':
        sample = weighted_random_sampling(population, n_samples)
    else:
        logging.info('No sampling type specified for {}, defaulting to simple random sample'.format(stock))
        sample = weighted_random_sampling(population, n_samples)
    return sample


def weighted_random_sampling(population, n_samples, weights=None):
    """
    :param population:
    :param n_samples:
    :param weights:
    :return samples:
    """
    if weights is not None:
        samples = population.sample(n=n_samples, replace=True, weights=weights)
    else:
        samples = population.sample(n=n_samples, replace=True)
    return samples


def sequential_sampling(population, overlap_array, n_samples, decay_rate=200):
    """

    :param population:
    :param overlap_array:
    :param n_samples:
    :param decay_rate:

    :return samples:
    """
    samples = pd.DataFrame(columns=population.columns)
    sample_weights = [1 / len(population)] * len(population)
    resampling_array = np.zeros(shape=(len(population), len(population)))
    decay = 1 / np.exp(-population.index / decay_rate)

    for sample in range(n_samples):
        sample = population.sample(n=1, weights=sample_weights)
        sample_date = pd.to_datetime(sample['DATE'].item())
        strike_date = pd.to_datetime(sample['STRIKE_DATE'].item())
        overlapping_labels = population[(population['DATE'] >= sample_date) & \
                                       (population['DATE'] <= strike_date)].index.tolist()
        resampling_array[min(overlapping_labels):max(overlapping_labels), :] += \
            np.where(overlap_array[min(overlapping_labels):max(overlapping_labels), :] == 1, 1, 0)
        sample_weights = np.divide(decay, (sum(resampling_array.T) + 1))
        sample_weights = sample_weights / sum(sample_weights)
        samples = samples.append(sample, ignore_index=True)

    return samples


def create_overlap_array(labeled_ts):

    overlap_array = np.zeros(shape=(len(labeled_ts), len(labeled_ts)))
    max_date = max(labeled_ts['DATE'])
    for ind, row in labeled_ts.iterrows():
        ind_buy = ind
        ind_sell = labeled_ts[labeled_ts['DATE'] == min(row['STRIKE_DATE'], max_date)].index.item()
        overlap_array[ind_buy:ind_sell + 1, ind] = 1

    return overlap_array


def create_uniqueness_variable(overlap_array):
    unique_keys = []
    for series in overlap_array.T:
        uniqueness = sum(series / (sum(overlap_array.T))) / sum(series)
        unique_keys.append(uniqueness)
    return unique_keys