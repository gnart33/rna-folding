# 1. Data Exploration
# 2. Data Preprocessing
#    - Sequence encoding
#    - Label grouping and padding (with NaN handling)
# 3. Model Building using a fast CNN architecture
# 4. Model Training with early stopping
# 5. Prediction on test set and submission file generation

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass


class DataLoader:
    """Data paths configuration."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.data = {}

    def load_data(self):
        all_csvs = list(self.root_path.glob("*.csv"))
        for csv in all_csvs:
            file_name = csv.stem
            self.data[file_name] = pd.read_csv(csv)

    def get_data(self, name: str) -> pd.DataFrame:
        return self.data[name]

    def get_data_names(self) -> List:
        return list(self.data.keys())


class DataPreprocessor:
    """Handles data preprocessing and preparation."""

    def __init__(self, max_length: int = 457):
        self.max_length = max_length
        self.nucleotide_map = {"A": 1, "C": 2, "G": 3, "U": 4}

    def encode_sequence(self, sequence: str) -> List[int]:
        """Convert RNA sequence to numerical encoding."""
        return [self.nucleotide_map[nuc] for nuc in sequence]

    def prepare_data(
        self,
        train_sequences: pd.DataFrame,
        valid_sequences: pd.DataFrame,
        test_sequences: pd.DataFrame,
        train_labels: pd.DataFrame,
        valid_labels: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare all data for model training."""
        # Fill missing values
        train_labels.fillna(0, inplace=True)
        valid_labels.fillna(0, inplace=True)

        # Encode sequences
        train_encoded = [
            self.encode_sequence(seq) for seq in train_sequences["sequence"]
        ]
        valid_encoded = [
            self.encode_sequence(seq) for seq in valid_sequences["sequence"]
        ]
        test_encoded = [self.encode_sequence(seq) for seq in test_sequences["sequence"]]

        # Pad sequences
        X_train = pad_sequences(train_encoded, maxlen=self.max_length, padding="post")
        X_valid = pad_sequences(valid_encoded, maxlen=self.max_length, padding="post")
        X_test = pad_sequences(test_encoded, maxlen=self.max_length, padding="post")

        # Prepare target variables
        y_train = train_labels["target"].values
        y_valid = valid_labels["target"].values

        return X_train, X_valid, X_test, y_train, y_valid


class RNAFoldingModel:
    """Handles model building and training."""

    def __init__(self, input_length: int, embedding_dim: int = 32):
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = None

    def build_model(self) -> Model:
        """Build the RNA folding prediction model."""
        # Input layer
        input_layer = Input(shape=(self.input_length,))

        # Embedding layer
        x = Embedding(input_dim=5, output_dim=self.embedding_dim)(input_layer)

        # Convolutional layers
        x = Conv1D(filters=64, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=128, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Output layer
        output_layer = Conv1D(filters=1, kernel_size=1, activation="sigmoid")(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping."""
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping],
            verbose=1,
        )

        return self.history.history

    def plot_training_history(self, save_path: str = "training_history.png") -> None:
        """Plot and save training history."""
        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["accuracy"], label="Training Accuracy")
        plt.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_model(self, path: str = "rna_folding_model.h5") -> None:
        """Save the trained model."""
        self.model.save(path)


class Predictor:
    """Handles model predictions and submission file generation."""

    def __init__(self, model: Model, sample_submission: pd.DataFrame):
        self.model = model
        self.sample_submission = sample_submission

    def generate_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions for test set."""
        return self.model.predict(X_test)

    def create_submission(
        self, predictions: np.ndarray, output_path: str = "submission.csv"
    ) -> None:
        """Create and save submission file."""
        submission = self.sample_submission.copy()
        submission["target"] = predictions.flatten()
        submission.to_csv(output_path, index=False)


def main():
    """Main function to run the RNA folding prediction pipeline."""
    # Set random seed
    np.random.seed(42)
    tf.random.set_seed(42)

    # Initialize data paths
    paths = DataPaths()

    # Data Exploration
    explorer = DataExplorer(paths)
    explorer.load_data()
    explorer.explore_data()

    # Data Preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_valid, X_test, y_train, y_valid = preprocessor.prepare_data(
        explorer.train_sequences,
        explorer.valid_sequences,
        explorer.test_sequences,
        explorer.train_labels,
        explorer.valid_labels,
    )

    # Model Building and Training
    model = RNAFoldingModel(input_length=X_train.shape[1])
    model.build_model()
    model.train(X_train, y_train, X_valid, y_valid)
    model.plot_training_history()
    model.save_model()

    # Generate Predictions
    predictor = Predictor(model.model, explorer.sample_submission)
    predictions = predictor.generate_predictions(X_test)
    predictor.create_submission(predictions)


if __name__ == "__main__":
    main()
