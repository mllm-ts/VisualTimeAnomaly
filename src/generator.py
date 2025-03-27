import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from tqdm import trange
from utils import plot_series, vector_to_interval, vector_to_point, vector_to_id, plot_rectangle_stack_series
import pickle
import math
from scipy.signal import square, sawtooth
from scipy.io import arff

def triangle_wave(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = 2 * np.abs((timestamp * freq) % 1 - 0.5) - 1 
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

def square_wave(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = square(2 * np.pi * freq * timestamp) 
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

def sawtooth_wave(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = sawtooth(2 * np.pi * freq * timestamp) 
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

def random_walk(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    steps = np.random.normal(0, noise_amp, length) 
    value = np.cumsum(steps)  
    value = coef * value + offset
    return value

def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

def cosine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = np.cos(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

class MultivariateDataGenerator:
    def __init__(self, data_dir, dim, drop_ratio=0):
        STREAM_LENGTH = 200

        self.stream_length = STREAM_LENGTH
        self.behavior = [sine, cosine]
        self.ano_behavior = {
            'triangle': triangle_wave,
            'square': square_wave,
            'sawtooth': sawtooth_wave,
            'random_walk': random_walk
        }
        self.dim = dim

        self.data = None
        self.label = None
        self.data_origin = None

        self.drop_ratio = drop_ratio
        self.data_dir = data_dir
        self.series = []
        self.anom = []

    def generate(self, num_ts, category, anomaly_type, train_eval, tsname):
        for i in trange(num_ts):
            self.generate_base_timeseries(category, tsname)
            self.variate_outliers(anomaly_type)

            if scenario == 'irr_multivariate':
                for dim_id in range(self.dim):
                    self.data[:, dim_id], self.label[:, dim_id], _ = drop(self.data[:, dim_id], self.label[:, dim_id], self.drop_ratio)

            anom = vector_to_id(self.label)

            self.series.append(self.data)
            self.anom.append(anom)

            fig = plot_rectangle_stack_series(
                series=self.data, 
                single_series_figsize=(10, 10),
                gt_anomaly=anom,
                train_eval = train_eval
            )

            fig_dir = os.path.join(self.data_dir, 'fig')
            os.makedirs(fig_dir, exist_ok=True)

            fig_path = os.path.join(fig_dir, f'{i + 1:03d}.png')
            fig.savefig(fig_path)
            plt.close()

        self.save()

    def save(self):
        data_dict = {
            'series': self.series,
            'anom': self.anom
        }
        with open(os.path.join(self.data_dir, 'data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)

    def generate_random_config(self):
        # Generate random parameters for time series using np.random
        return {
            'freq': np.random.uniform(0.03, 0.05),  # Frequency between 0.01 and 0.1
            'coef': np.random.uniform(0.5, 2.0),  # Coefficient between 0.5 and 2.0
            'offset': np.random.uniform(-1.0, 1.0),  # Offset between -1.0 and 1.0
            'noise_amp': np.random.uniform(0.05, 0.20),  # Noise amplitude between 0.0 and 0.1
            'length': self.stream_length  # Length of the time series
        }

    def generate_base_timeseries(self, category, basedata_dir=None):
        if category == 'synthetic':
            self.data = np.zeros((self.stream_length, self.dim))
            for i in range(self.dim):
                behavior = np.random.choice(self.behavior)
                config = self.generate_random_config()
                uni_data = behavior(**config)
                self.data[:, i] = uni_data
            self.data_origin = self.data.copy()
            self.label = np.zeros((self.stream_length, self.dim), dtype=float)
        elif category == 'semi':
            basedata_dir = f'Multivariate_arff/{tsname}/{tsname}_TEST.arff'
            raw_data = arff.loadarff(basedata_dir)
            df = pd.DataFrame(raw_data[0])
            self.data = np.array([list(item) for item in df.iloc[0,0]]).transpose()
            self.stream_length, dim = self.data.shape
            if self.dim > dim:
                extra_dims = self.dim - dim
                repeat_indices = np.random.choice(dim, extra_dims, replace=True)
                self.data = np.hstack((self.data, self.data[:, repeat_indices]))
            elif self.dim < dim:
                selected_indices = np.random.choice(dim, self.dim, replace=False)
                self.data = self.data[:, selected_indices]
            self.data_origin = self.data.copy()
            self.label = np.zeros((self.stream_length, self.dim), dtype=float)
    
    def variate_outliers(self, anomaly_type):
        min_ano, max_ano = 1, math.floor(math.sqrt(self.dim)) - 1
        num_anomalies = np.random.randint(min_ano, max_ano + 1)
        anomaly_indices = np.random.choice(self.dim, num_anomalies, replace=False)

        for idx in anomaly_indices:
            ano_behavior = self.ano_behavior[anomaly_type]
            config = self.generate_random_config()
            anomaly_data = ano_behavior(**config)  
            self.data[:, idx] = anomaly_data
            self.label[:, idx] = 1  # Mark this variate as an anomaly


def drop(data, label, drop_ratio):
    if not 0 <= drop_ratio <= 1:
        raise ValueError("drop_ratio must be between 0 and 1.")
    
    seq_len = len(data)
    num_drops = int(seq_len * drop_ratio)
    
    # Generate random indices to drop
    drop_index = np.random.choice(seq_len, size=num_drops, replace=False)
    
    data = data.astype(float)  # Ensure float type to allow np.nan
    label = label.astype(float)  # Ensure float type to allow np.nan

    data[drop_index] = np.nan
    label[drop_index] = np.nan

    return data, label, drop_index

def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value

def collective_global_synthetic(length, base, coef=1.5, noise_amp=0.005):
    value = []
    norm = np.linalg.norm(base)
    base = base / norm
    num = int(length / len(base))
    for i in range(num):
        value.extend(base)
    residual = length - len(value)
    value.extend(base[:residual])
    value = np.array(value)
    noise = np.random.normal(0, 1, length)
    value = coef * value + noise_amp * noise
    return value

# The code is adapted from https://github.com/datamllab/tods/tree/benchmark
class UnivariateDataGenerator:
    def __init__(self, data_dir, drop_ratio=0):
        BEHAVIOR = sine
        BEHAVIOR_CONFIG = {'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05}
        STREAM_LENGTH = 400

        self.behavior = BEHAVIOR
        self.behavior_config = BEHAVIOR_CONFIG
        self.stream_length = STREAM_LENGTH

        self.data = None
        self.label = None
        self.data_origin = None

        self.drop_ratio = drop_ratio
        self.data_dir = data_dir
        self.series = []
        self.anom = []
        self.drop_index = []

    def generate(self, num_ts, category, anomaly_type, train_eval, tsname):
        for i in trange(num_ts):
            self.generate_base_timeseries(category, tsname)

            if anomaly_type == 'global':
                self.point_global_outliers(ratio=0.05, factor=3.5, radius=5)
            elif anomaly_type == 'contextual':
                self.point_contextual_outliers(ratio=0.05, factor=2.5, radius=5)
            elif anomaly_type == 'seasonal':
                self.collective_seasonal_outliers(ratio=0.05, factor=3, radius=5)
            elif anomaly_type == 'trend':
                self.collective_trend_outliers(ratio=0.05, factor=0.5, radius=5)
            elif anomaly_type == 'shapelet':
                self.collective_global_outliers(ratio=0.05, radius=5, option='square', coef=1.5, noise_amp=0.03, level=20, freq=0.04, offset=0.0)

            if scenario == 'irr_univariate':
                self.data, self.label, drop_index = drop(self.data, self.label, self.drop_ratio)
            
            if anomaly_type in ['global', 'contextual']:
                anom = vector_to_point(self.label)
            else:
                anom = vector_to_interval(self.label)
            
            self.series.append(self.data)
            self.anom.append(anom)

            if scenario == 'irr_univariate':
                self.drop_index.append(drop_index)

            fig = plot_series(
                series=self.data,
                single_series_figsize=(10, 1.5),
                gt_anomaly=anom,
                # gt_ylim = (np.nanmin(self.data)*1.1, np.nanmax(self.data)*1.1),
                train_eval = train_eval
            )

            fig_dir = os.path.join(self.data_dir, 'fig')
            os.makedirs(fig_dir, exist_ok=True)

            fig_path = os.path.join(fig_dir, f'{i + 1:03d}.png')
            fig.savefig(fig_path)
            plt.close()

        self.save()

    def save(self):
        data_dict = {
            'series': self.series,
            'anom': self.anom
        }
        if scenario == 'irr_univariate':
            data_dict['drop_index'] = self.drop_index
        with open(os.path.join(self.data_dir, 'data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        
    def generate_base_timeseries(self, category, tsname, basedata_dir=None):
        if category == 'synthetic':
            self.behavior_config['length'] = self.stream_length
            self.data = self.behavior(**self.behavior_config)
            self.data_origin = self.data.copy()
            self.label = np.zeros(self.stream_length, dtype=float)
        elif category == 'semi':
            basedata_dir = f'Univariate_arff/{tsname}/{tsname}_TEST.arff'
            raw_data = arff.loadarff(basedata_dir)
            df = pd.DataFrame(raw_data[0])
            self.data = df.iloc[0, :-1].values
            self.stream_length = self.data.shape[0]
            self.data_origin = self.data.copy()
            self.label = np.zeros(self.stream_length, dtype=float)

    def point_global_outliers(self, ratio, factor, radius):
        """
        Add point global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.stream_length * ratio)) * self.stream_length).astype(int)
        maximum, minimum = max(self.data), min(self.data)
        for i in position:
            local_std = self.data_origin[max(0, i - radius):min(i + radius, self.stream_length)].std()
            self.data[i] = self.data_origin[i] * factor * local_std
            if 0 <= self.data[i] < maximum: self.data[i] = maximum
            if 0 > self.data[i] > minimum: self.data[i] = minimum
            self.label[i] = 1
        
    def point_contextual_outliers(self, ratio, factor, radius):
        """
        Add point contextual outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
                    Notice: point contextual outliers will not exceed the range of [min, max] of original data
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.stream_length * ratio)) * self.stream_length).astype(int)
        maximum, minimum = max(self.data), min(self.data)
        for i in position:
            local_std = self.data_origin[max(0, i - radius):min(i + radius, self.stream_length)].std()
            self.data[i] = self.data_origin[i] * factor * local_std
            if self.data[i] > maximum: self.data[i] = maximum * min(0.95, abs(np.random.normal(0, 0.5)))  # previous(0, 1)
            if self.data[i] < minimum: self.data[i] = minimum * min(0.95, abs(np.random.normal(0, 0.5)))

            self.label[i] = 1
        
    def collective_global_outliers(self, ratio, radius, option='square', coef=3., noise_amp=0.0,
                                    level=5, freq=0.04, offset=0.0, # only used when option=='square'
                                    base=[0.,]): # only used when option=='other'
        """
        Add collective global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            radius: the radius of collective outliers range
            option: if 'square': 'level' 'freq' and 'offset' are used to generate square sine wave
                    if 'other': 'base' is used to generate outlier shape
            level: how many sine waves will square_wave synthesis
            base: a list of values that we want to substitute inliers when we generate outliers
        """
        base = [1.4529900e-01, 1.2820500e-01, 9.4017000e-02, 7.6923000e-02, 1.1111100e-01, 1.4529900e-01, 1.7948700e-01, 2.1367500e-01, 2.1367500e-01]
        position = (np.random.rand(round(self.stream_length * ratio / (2 * radius))) * self.stream_length).astype(int)

        valid_option = {'square', 'other'}
        if option not in valid_option:
            raise ValueError("'option' must be one of %r." % valid_option)

        if option == 'square':
            sub_data = square_sine(level=level, length=self.stream_length, freq=freq,
                                   coef=coef, offset=offset, noise_amp=noise_amp)
        else:
            sub_data = collective_global_synthetic(length=self.stream_length, base=base,
                                                   coef=coef, noise_amp=noise_amp)
        for i in position:
            start, end = max(0, i - radius), min(self.stream_length, i + radius)
            self.data[start:end] = sub_data[start:end]
            self.label[start:end] = 1
        
    def collective_trend_outliers(self, ratio, factor, radius):
        """
        Add collective trend outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how dramatic will the trend be
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.stream_length * ratio / (2 * radius))) * self.stream_length).astype(int)
        for i in position:
            start, end = max(0, i - radius), min(self.stream_length, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[start:end] = self.data_origin[start:end] + slope
            self.data[end:] = self.data[end:] + slope[-1]
            self.label[start:end] = 1
        
    def collective_seasonal_outliers(self, ratio, factor, radius):
        """
        Add collective seasonal outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how many times will frequency multiple
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.stream_length * ratio / (2 * radius))) * self.stream_length).astype(int)
        seasonal_config = self.behavior_config
        seasonal_config['freq'] = factor * self.behavior_config['freq']
        for i in position:
            start, end = max(0, i - radius), min(self.stream_length, i + radius)
            self.data[start:end] = self.behavior(**seasonal_config)[start:end]
            self.label[start:end] = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--category", type=str, default='synthetic', choices=['synthetic', 'semi'])
    parser.add_argument("--scenario", type=str, default='univariate', choices=['univariate', 'multivariate', 'irr_univariate', 'irr_multivariate'])
    parser.add_argument("--tsname", type=str, default=None, help='base real world dataset name')
    parser.add_argument("--anomaly_type", type=str, default='global', choices=['global', 'contextual', 'seasonal', 'trend', 'shapelet', 
                                                                               'triangle', 'square', 'sawtooth',  'random_walk',
                                                                               'long'])
    parser.add_argument("--num_ts", type=int, default=100, help="Numebr of time series")
    parser.add_argument("--dim", type=int, default=9, help="Number of variates of multivariate time series") # min_dim=4
    parser.add_argument("--drop_ratio", type=float, default=0.00, help="Dropping ratio of irregular time series")

    args = parser.parse_args()
    
    seed = args.seed
    category = args.category
    scenario = args.scenario
    tsname = args.tsname
    anomaly_type = args.anomaly_type
    num_ts = args.num_ts
    dim = args.dim
    drop_ratio = args.drop_ratio

    np.random.seed(seed)

    for train_eval in ['train', 'eval']:
        if category == 'synthetic':
            if scenario == 'univariate':
                data_dir = os.path.join('data', category, scenario, anomaly_type, train_eval)
            elif scenario == 'multivariate':
                data_dir = os.path.join('data', category, scenario, f'dim_{dim}', anomaly_type, train_eval)
            elif scenario.startswith('irr'):
                data_dir = os.path.join('data', category, scenario, f'ratio_{int(drop_ratio*100)}', anomaly_type, train_eval)
        elif category == 'semi':
            if scenario == 'univariate':
                data_dir = os.path.join('data', category, scenario, tsname, anomaly_type, train_eval)
            elif scenario == 'multivariate':
                data_dir = os.path.join('data', category, scenario, tsname, f'dim_{dim}', anomaly_type, train_eval)
            elif scenario.startswith('irr'):
                data_dir = os.path.join('data', category, scenario, tsname, f'ratio_{int(drop_ratio*100)}', anomaly_type, train_eval)

        print(f'Generating {data_dir} data.')

        if scenario.endswith('univariate'):
            univariate_generator = UnivariateDataGenerator(data_dir=data_dir, drop_ratio=drop_ratio)
            univariate_generator.generate(num_ts, category, anomaly_type, train_eval, tsname)
        elif scenario.endswith('multivariate'):
            multivariate_generator = MultivariateDataGenerator(data_dir=data_dir, dim=dim, drop_ratio=drop_ratio)
            multivariate_generator.generate(num_ts, category, anomaly_type, train_eval, tsname)
