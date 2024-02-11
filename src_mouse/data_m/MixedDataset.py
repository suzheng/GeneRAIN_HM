import torch
import random
from torch.utils.data import Dataset

class MixedDataset(Dataset):
    @staticmethod
    def set_seed(seed):
        random.seed(seed)

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_size = len(dataset1) + len(dataset2)

        # Create a list of indices for dataset1 and dataset2
        self.indices = [(i, 1) for i in range(len(dataset1))] + [(i, 2) for i in range(len(dataset2))]
        random.shuffle(self.indices)  # Shuffle the indices to mix the datasets

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        original_idx, dataset_number = self.indices[idx]

        if dataset_number == 1:
            return self.dataset1[original_idx]
        else:
            return self.dataset2[original_idx]