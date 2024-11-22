import torch
import torch.nn as nn

class DisasterModel(nn.Module):
    def __init__(self):
        super(DisasterModel, self).__init__()
        # Define layers (Siamese, Encoder-Decoder)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256*256*64, 2)  # Example size after flattening (adjust based on image size)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
