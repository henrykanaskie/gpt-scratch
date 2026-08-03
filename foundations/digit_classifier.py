import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.input_dim = 784
        self.hidden_dim = 512
        self.output_dim = 10
        self.dropout = .2
        self.model = nn.Sequential(nn.Linear(self.input_dim, self. hidden_dim), nn.ReLU(), nn.Dropout(p=self.dropout), nn.Linear(self.hidden_dim, self.output_dim), nn.Sigmoid())
        

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.model(images)
        return torch.round(x, decimals=4)
        
