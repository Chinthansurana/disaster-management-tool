import torch
import torch.nn as nn
import torch.nn.functional as F

# Siamese Neural Network + Encoder-Decoder for Damage Detection

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return torch.abs(output1 - output2)

class EncoderDecoderDamageDetection(nn.Module):
    def __init__(self):
        super(EncoderDecoderDamageDetection, self).__init__()
        # Encoder (Convolutional Layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Decoder (Upsampling Layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
