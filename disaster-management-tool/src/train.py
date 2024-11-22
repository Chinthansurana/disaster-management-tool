import torch
import torch.nn as nn
from utils.config_loader import load_config
from model import DisasterModel  # Your model class (Siamese, Encoder-Decoder)
from torch.utils.data import DataLoader
import os

# Load config
config = load_config("config.yaml")

# Training parameters
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']

# Example model training loop
def train_model():
    model = DisasterModel()
    criterion = nn.CrossEntropyLoss()  # or other loss functions based on your setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Assuming dataset is loaded (use DataLoader with your dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), config['model']['save_path'])

if __name__ == "__main__":
    train_model()
