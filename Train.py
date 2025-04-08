import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import numpy as np 
import pickle

from Data_Loader import TSPDataset
from PtrNetwork import PtrNetwork

if __name__ == '__main__':
    
    # HYPERPARAMS
    npoints = 10
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
    
    
    # data load
    data_file_path = 'data/tsp_10_test_exact.txt'
        
    dataset = TSPDataset(data_file_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)




    train_losses = []
    val_losses = []

    # Train

    best_val_loss = float('inf')
    save_path = "best_model.pt"

    model = PtrNetwork(INPUT_SIZE, HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # padding_mask = torch.ones(BATCH_SIZE, SEQ_LEN)


    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        total_loss = 0.0
        batch_cnt = 0

        for cities, answer in train_loader:
            optimizer.zero_grad()
            # loss = model(cities, target_len=SEQ_LEN, padding_mask = padding_mask, targets=answer)
            loss = model(cities, target_len=SEQ_LEN, targets=answer)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_cnt += 1
        
        avg_loss = total_loss / batch_cnt
        # print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
        # model validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for v_cities, v_answer in val_loader:
                val_loss = model(v_cities, target_len=SEQ_LEN, targets= v_answer)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Validation Loss : {avg_val_loss:.4f}")
        
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model updated and saved (val loss = {best_val_loss:.4f})")
            
            
            
    ## Save loss log
    with open("loss_log.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i in range(EPOCHS):
            f.write(f"{i+1},{train_losses[i]},{val_losses[i]}\n")
