import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

# time series image dataset
class TSIDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.figs_dir = os.path.join(data_dir, 'figs')
        self.series = []
        self.anom = []

        # Load data
        with open(os.path.join(self.data_dir, 'data.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        self.series = data_dict['series']
        self.anom = data_dict['anom']
        self.scenario = data_dir.split('/')[2]
        train_eval = data_dir.split('/')[-1]

        if self.scenario == 'irr_univariate':
            self.drop_index = data_dict['drop_index']
        if train_eval == 'eval':
            print(f"Loaded dataset {data_dir} with {len(self.series)} series.")

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        anom = self.anom[idx]
        series = self.series[idx]

        # Convert to torch tensors
        anom = torch.tensor(anom, dtype=torch.float32)
        series = torch.tensor(series, dtype=torch.float32)
        if self.scenario == 'irr_univariate':
            drop_index = self.drop_index[idx]
            drop_index = torch.tensor(drop_index, dtype=torch.float32)
            return anom, series, drop_index
        else:
            return anom, series
        
    def few_shots(self, num_shots=5, idx=None):
        if idx is None:
            idx = np.random.choice(len(self.series), num_shots, replace=False)
        few_shot_data = []
        for i in idx:
            anom, series = self.__getitem__(i)
            anom = [{"start": int(start.item()), "end": int(end.item())} for start, end in list(anom[0])]
            few_shot_data.append((series, anom, i+1))
        return few_shot_data