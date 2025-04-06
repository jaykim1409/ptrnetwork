import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np 
import pickle

from Data_Loader import TSPDataset
from PtrNetwork import PtrNetwork



# data load

with open('X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('dist_info.pkl', 'rb') as f:
    dist_info = pickle.load(f)
with open('y.pkl', 'rb') as f:
    y = pickle.load(f)
    
dataset = TSPDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# HYPERPARAMS
npoints = 20
SEQ_LEN = npoints
INPUT_SIZE = 2
OUTPUT_SIZE = 10
HIDDEN_SIZE = 256
LR = 0.001
BATCH_SIZE = 128
ENC_SEQ_LEN = npoints
DEC_SEQ_LEN = npoints
GRAD_CLIP = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50



# Train

model = PtrNetwork(INPUT_SIZE, HIDDEN_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)

padding_mask = torch.ones(BATCH_SIZE, SEQ_LEN)


for epoch in range(1, EPOCHS + 1):
    model.train()
    
    total_loss = 0.0
    batch_cnt = 0

    for cities, answer in dataloader:
        optimizer.zero_grad()
        loss = model(cities, target_len=SEQ_LEN, padding_mask = padding_mask, targets=answer)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_cnt += 1
    
    avg_loss = total_loss / batch_cnt
    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4f}")