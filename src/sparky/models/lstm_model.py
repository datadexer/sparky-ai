"""LSTM sequence model for crypto signal prediction.

Implements ModelProtocol for backtester compatibility. Converts tabular
features into rolling windows (sequences) and trains a 2-layer LSTM with
early stopping for robust time-series prediction.

Architecture:
- Input: (batch, window, n_features) rolling windows of tabular data
- LSTM: 2 layers (input→64→32) with dropout=0.2 between layers
- FC: 32→1 sigmoid output (binary classification)
- Output: {0=flat, 1=long} signals

Training:
- Early stopping: monitors validation loss with patience=10
- Max epochs: 100 (typical convergence: 30-50)
- Train/val split: 80/20 on training data
- Optimizer: Adam with learning_rate=0.001
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """2-layer LSTM network for binary classification.

    Args:
        input_size: Number of features per timestep.
        hidden_size_1: Hidden size of first LSTM layer (default 64).
        hidden_size_2: Hidden size of second LSTM layer (default 32).
        dropout: Dropout rate between LSTM layers (default 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.fc = nn.Linear(hidden_size_2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch, 1) with probabilities [0, 1].
        """
        # LSTM 1
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)

        # LSTM 2
        lstm2_out, _ = self.lstm2(lstm1_out)

        # Take last timestep's hidden state
        last_hidden = lstm2_out[:, -1, :]

        # FC layer + sigmoid
        out = self.fc(last_hidden)
        out = self.sigmoid(out)

        return out


class LSTMModel:
    """LSTM sequence model for crypto signal prediction.

    Implements ModelProtocol (fit/predict) for backtester compatibility.
    Converts tabular features to rolling window sequences and trains a
    2-layer LSTM with early stopping.

    Args:
        window_length: Rolling window size (default 10 days).
        hidden_size_1: First LSTM layer hidden size (default 64).
        hidden_size_2: Second LSTM layer hidden size (default 32).
        dropout: Dropout rate (default 0.2).
        learning_rate: Adam optimizer learning rate (default 0.001).
        max_epochs: Maximum training epochs (default 100).
        patience: Early stopping patience (default 10).
        val_split: Validation set fraction (default 0.2).
        batch_size: Training batch size (default 32).
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        window_length: int = 10,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        patience: int = 10,
        val_split: float = 0.2,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.window_length = window_length
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _create_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert tabular data to rolling window sequences.

        Given X of shape (n_samples, n_features), creates rolling windows
        of length `window_length` resulting in:
            X_seq: (n_sequences, window_length, n_features)
            y_seq: (n_sequences,) where y_seq[i] is the label for the
                   timestep AFTER the i-th window.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (n_samples,).

        Returns:
            Tuple of (X_sequences, y_sequences).
        """
        X_np = X.values
        y_np = y.values

        n_samples, n_features = X_np.shape

        # Need at least window_length + 1 samples to create one sequence
        if n_samples < self.window_length + 1:
            logger.warning(
                f"Insufficient samples ({n_samples}) for window_length={self.window_length}. "
                "Returning empty sequences."
            )
            return np.array([]), np.array([])

        # Create rolling windows
        X_seq = []
        y_seq = []

        for i in range(n_samples - self.window_length):
            # Window: X[i:i+window_length]
            # Target: y[i+window_length] (next timestep after window)
            window = X_np[i : i + self.window_length]
            target = y_np[i + self.window_length]

            X_seq.append(window)
            y_seq.append(target)

        X_seq = np.array(X_seq)  # Shape: (n_sequences, window_length, n_features)
        y_seq = np.array(y_seq)  # Shape: (n_sequences,)

        return X_seq, y_seq

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train LSTM model on features X and binary labels y.

        Converts tabular X to sequences, normalizes features, splits
        train/val, and trains with early stopping.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels {0, 1} where 1=long, 0=flat.
        """
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        if len(X_seq) == 0:
            logger.error("No sequences created — insufficient data for LSTM training")
            return

        # Normalize features (fit scaler on all training data)
        n_seq, window, n_feat = X_seq.shape
        X_flat = X_seq.reshape(-1, n_feat)
        X_scaled = self.scaler.fit_transform(X_flat).reshape(n_seq, window, n_feat)

        # Train/val split
        n_val = int(len(X_seq) * self.val_split)
        n_train = len(X_seq) - n_val

        X_train = X_scaled[:n_train]
        y_train = y_seq[:n_train]
        X_val = X_scaled[n_train:]
        y_val = y_seq[n_train:]

        logger.info(
            f"LSTM training: {n_train} train sequences, {n_val} val sequences, "
            f"window={self.window_length}, features={n_feat}"
        )

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Initialize model
        self.model = LSTMNet(
            input_size=n_feat,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Early stopping setup
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(self.max_epochs):
            self.model.train()

            # Mini-batch training
            n_batches = (n_train + self.batch_size - 1) // self.batch_size
            train_losses = []

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_train)

                X_batch = X_train_t[start:end]
                y_batch = y_train_t[start:end]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            mean_train_loss = np.mean(train_losses)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs}: "
                    f"train_loss={mean_train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        logger.info(f"LSTM training complete (best val_loss={best_val_loss:.4f})")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary signals {0, 1} for features X.

        Converts X to sequences, normalizes, and predicts using trained LSTM.
        Uses 0.5 probability threshold for classification.

        Edge case: If test set length < window_length, pad predictions with 0.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Binary signals as numpy array {0=flat, 1=long}.
        """
        if self.model is None:
            logger.warning("Model not trained yet — returning all zeros")
            return np.zeros(len(X), dtype=int)

        # Handle edge case: test set too short for window
        if len(X) < self.window_length:
            logger.warning(
                f"Test set too short ({len(X)} < {self.window_length}). "
                "Padding predictions with 0 (flat)."
            )
            return np.zeros(len(X), dtype=int)

        # Create sequences for prediction
        # Note: For prediction, we don't need labels, but _create_sequences expects them
        # Use dummy labels (all zeros) since we only care about X sequences
        dummy_y = pd.Series(np.zeros(len(X)), index=X.index)
        X_seq, _ = self._create_sequences(X, dummy_y)

        if len(X_seq) == 0:
            logger.warning("No sequences created for prediction — returning all zeros")
            return np.zeros(len(X), dtype=int)

        # Normalize using fitted scaler
        n_seq, window, n_feat = X_seq.shape
        X_flat = X_seq.reshape(-1, n_feat)
        X_scaled = self.scaler.transform(X_flat).reshape(n_seq, window, n_feat)

        # Convert to tensor and predict
        X_t = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy().flatten()

        # Threshold at 0.5
        predictions = (probs > 0.5).astype(int)

        # Pad predictions to match input length
        # First window_length samples have no prediction (used in sequences)
        # Prepend zeros for those
        padded_predictions = np.concatenate([
            np.zeros(self.window_length, dtype=int),
            predictions,
        ])

        return padded_predictions
