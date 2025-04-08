from torch.utils.data import Dataset, DataLoader
import torch

class TSPDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = list(map(float, parts[:20]))
                coords = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                tour = list(map(int, parts[21:-1]))
                tour = [x - 1 for x in tour]
                self.samples.append((coords, tour))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        coords, tour = self.samples[idx]
        coords_tensor = torch.tensor(coords, dtype = torch.float32)
        tour_tensor = torch.tensor(tour, dtype=torch.long)
        return coords_tensor, tour_tensor