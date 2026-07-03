"""PyTorch model definitions for ABSA task."""

import torch
import torch.nn as nn
from typing import List


class AspectClassifier(nn.Module):
    """Multi-class aspect classifier on top of DistilBERT embeddings."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_aspects: int = 8,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        l2_reg: float = 0.001
    ):
        """
        Args:
            embedding_dim: Input embedding dimension (DistilBERT = 768)
            num_aspects: Number of aspect classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            l2_reg: L2 regularization coefficient
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.l2_reg = l2_reg
        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_aspects))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns logits (before softmax)."""
        return self.network(x)

    def get_l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss."""
        l2_loss = torch.tensor(0.0, device=self.network[0].weight.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param) ** 2
        return self.l2_reg * l2_loss


class SentimentClassifier(nn.Module):
    """Binary sentiment classifier on top of DistilBERT embeddings."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        l2_reg: float = 0.001
    ):
        """
        Args:
            embedding_dim: Input embedding dimension (DistilBERT = 768)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            l2_reg: L2 regularization coefficient
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.l2_reg = l2_reg
        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Binary output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns logits (before sigmoid)."""
        return self.network(x)

    def get_l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss."""
        l2_loss = torch.tensor(0.0, device=self.network[0].weight.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param) ** 2
        return self.l2_reg * l2_loss
