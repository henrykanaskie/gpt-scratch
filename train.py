import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = 0
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start = torch.randint(high=(len(data) - context_length), size = (batch_size,))
            X = torch.stack([data[i:i + context_length] for i in start])
            Y = torch.stack([data[i+1:i+1 + context_length] for i in start])
            logits = model(X)
            b, t, v = logits.shape
            logits = torch.reshape(logits, (b * t, v))
            targets = torch.reshape(Y, (b * t,))
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)
