
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from causal.data_utils import ExposureDataset, build_exposure_dataset
import logging

def train_exposure_model(exposure_model, interactions_df, item_pool, device, epochs=5, batch_size=1024, lr=0.001):
    """
    Trains the Exposure Model explicitly using BCE Loss.
    """
    logging.info("--- [Causal] Training Exposure Model ---")
    
    # 1. Build Dataset
    raw_data = build_exposure_dataset(interactions_df, item_pool)
    dataset = ExposureDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(exposure_model.parameters(), lr=lr)
    exposure_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            user = batch['user_id'].to(device)
            item = batch['item_id'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Predict Propensity
            # ExposureModel.forward returns propensity (sigmoid applied)
            propensity = exposure_model(user, item)
            
            # BCE Loss
            loss = F.binary_cross_entropy(propensity.squeeze(), label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logging.info(f"[Exposure] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    logging.info("--- [Causal] Exposure Model Trained ---")
    exposure_model.eval()
    return exposure_model
