"""Training loop for aspect and sentiment classifiers."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict

class ModelTrainer:
    """Trainer for PyTorch models with early stopping and LR scheduling."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        device: str = None
    ):
        """
        Args:
            model: PyTorch model to train
            learning_rate: Initial learning rate
            weight_decay: L2 penalty (standard PyTorch weight decay)
            device: Device to train on
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        self.best_val_loss = float('inf')

    def train_aspect_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Train multi-class aspect classifier.

        Args:
            X_train, y_train: Training embeddings and labels
            X_val, y_val: Validation embeddings and labels
            epochs: Maximum epochs
            batch_size: Batch size
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Print progress

        Returns:
            Dictionary with train/val history
        """
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
        y_val_tensor = torch.from_numpy(y_val).long().to(self.device)

        criterion = nn.CrossEntropyLoss()
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)

                if hasattr(self.model, 'get_l2_loss'):
                    loss = loss + self.model.get_l2_loss()

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                train_correct += (logits.argmax(1) == y_batch).sum().item()
                train_total += X_batch.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_correct = (val_logits.argmax(1) == y_val_tensor).sum().item()
                val_acc = val_correct / len(y_val_tensor)

            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)

            self.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_state)
                    break

        return {'train': self.train_history, 'val': self.val_history}

    def train_sentiment_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Train binary sentiment classifier.

        Args:
            X_train, y_train: Training embeddings and binary labels
            X_val, y_val: Validation embeddings and binary labels
            epochs: Maximum epochs
            batch_size: Batch size
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Print progress

        Returns:
            Dictionary with train/val history
        """
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float().unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
        y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)

                if hasattr(self.model, 'get_l2_loss'):
                    loss = loss + self.model.get_l2_loss()

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                pred = (torch.sigmoid(logits) > 0.5).long()
                train_correct += (pred == y_batch.long()).sum().item()
                train_total += X_batch.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_pred = (torch.sigmoid(val_logits) > 0.5).long()
                val_correct = (val_pred == y_val_tensor.long()).sum().item()
                val_acc = val_correct / len(y_val_tensor)

            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)

            self.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_state)
                    break

        return {'train': self.train_history, 'val': self.val_history}

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions on embeddings."""
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            predictions = []
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i : i + batch_size]
                logits = self.model(batch)
                predictions.append(logits.cpu().numpy())

        return np.vstack(predictions)
