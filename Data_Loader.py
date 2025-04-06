from torch.utils.data import Dataset, DataLoader
import torch


class TSPDataset(Dataset):
    def __init__(self, data_dict, label_dict):
        self.data = []
        self.labels = []
        
        for key in data_dict.keys():
            self.data.append(torch.tensor(data_dict[key], dtype=torch.float32))   # shape: (20, 2)
            self.labels.append(torch.tensor(label_dict[key], dtype=torch.long))   # shape: (20,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
