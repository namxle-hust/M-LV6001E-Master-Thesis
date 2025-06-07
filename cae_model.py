import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse
import logging

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class OmicsDataset(Dataset):
    """Dataset class for omics data"""

    def __init__(self, data_path, transform=None):
        # Read the TSV file
        self.data = pd.read_csv(data_path, sep="\t", index_col=0)

        # Transpose to have samples as rows and features as columns
        self.data = self.data.T

        # Convert to numpy array
        self.data_array = self.data.values.astype(np.float32)

        # Apply Min-Max normalization as mentioned in the paper
        self.scaler = MinMaxScaler()
        self.data_array = self.scaler.fit_transform(self.data_array)

        self.transform = transform

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        sample = self.data_array[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.from_numpy(sample)


class ContractiveAutoEncoder(nn.Module):
    """
    Contractive AutoEncoder implementation based on the CtAE paper
    Architecture: input -> 5000 -> 128 -> 64 -> 128 -> 5000 -> output
    """

    def __init__(self, input_dim, hidden_dims=[5000, 128, 64], dropout_rate=0.2):
        super(ContractiveAutoEncoder, self).__init__()

        # Store dimensions for later use
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim, bias=True))
            encoder_layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # Don't add dropout to bottleneck layer
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        reversed_dims = hidden_dims[::-1]

        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(
                nn.Linear(reversed_dims[i], reversed_dims[i + 1], bias=True)
            )
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))

        # Final decoder layer with sigmoid activation
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim, bias=True))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

        # Store the first encoder layer weight for contractive loss
        self.encoder_weight = self.encoder[0].weight

    def forward(self, x):
        # Get hidden representation
        h = self.encoder(x)
        # Get reconstruction
        x_recon = self.decoder(h)
        return h, x_recon

    def get_encoder_output(self, x):
        """Get only the encoder output (latent representation)"""
        return self.encoder(x)


def contractive_loss(x, x_recon, h, W, lambda_reg=1e-4):
    """
    Compute the Contractive AutoEncoder Loss

    Args:
        x: original input
        x_recon: reconstructed input
        h: hidden representation from encoder
        W: weight matrix of the first encoder layer
        lambda_reg: regularization coefficient (default: 1e-4 as per paper)

    Returns:
        total_loss: reconstruction loss + contractive penalty
        recon_loss: reconstruction loss only
        contract_loss: contractive penalty only
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    # Contractive loss
    # Calculate the derivative of hidden layer with respect to input
    # For ReLU activation: dh/dx = W if h > 0, else 0
    # Since we used sequential layers, we need to get the actual hidden values
    # after the first linear layer but before ReLU

    # Get the output of first linear layer
    h_linear = (
        torch.matmul(x, W.t()) + W.bias
        if hasattr(W, "bias")
        else torch.matmul(x, W.t())
    )

    # Derivative of ReLU
    dh = (h_linear > 0).float()

    # Frobenius norm of Jacobian
    # J_ij = dh_i/dx_j = W_ij * dh_i
    # ||J||_F^2 = sum_ij (W_ij * dh_i)^2

    # Efficient computation as suggested in the paper
    w_squared = W**2
    contract_loss = torch.sum(
        torch.mm(dh**2, w_squared.sum(dim=1).unsqueeze(1))
    ) / x.size(0)

    # Total loss
    total_loss = recon_loss + lambda_reg * contract_loss

    return total_loss, recon_loss, contract_loss


def train_cae(
    model,
    train_loader,
    num_epochs=200,
    learning_rate=0.001,
    lambda_reg=0.0001,
    device="cuda",
    patience=20,
):
    """
    Train the Contractive AutoEncoder

    Args:
        model: ContractiveAutoEncoder instance
        train_loader: DataLoader for training data
        num_epochs: number of training epochs (default: 200 as per paper)
        learning_rate: learning rate (default: 0.001 as per paper)
        lambda_reg: regularization coefficient (default: 0.0001 as per paper)
        device: 'cuda' or 'cpu'
        patience: epochs to wait before early stopping
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    recon_losses = []
    contract_losses = []

    # Best model tracking
    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_contract_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()

            # Forward pass
            h, x_recon = model(data)

            # Calculate loss
            loss, recon_loss, contract_loss = contractive_loss(
                data, x_recon, h, model.encoder_weight, lambda_reg
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_contract_loss += contract_loss.item()

        # Average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_contract_loss = epoch_contract_loss / len(train_loader)

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        contract_losses.append(avg_contract_loss)

        # Check if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Loss: {avg_loss:.4f}, "
                f"Recon Loss: {avg_recon_loss:.4f}, "
                f"Contract Loss: {avg_contract_loss:.4f}"
            )

            if best_model_state is not None:
                logger.info(f"  Best Loss: {best_loss:.4f} at epoch {best_epoch+1}")

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            logger.info(f"Best model was at epoch {best_epoch+1} with loss {best_loss:.4f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored best model from epoch {best_epoch+1}")

    return train_losses, recon_losses, contract_losses, best_epoch


def extract_features(model, data_loader, device="cuda"):
    """
    Extract latent features from trained model

    Args:
        model: trained ContractiveAutoEncoder
        data_loader: DataLoader for data
        device: 'cuda' or 'cpu'

    Returns:
        features: numpy array of latent features
    """
    model.eval()
    model = model.to(device)
    features = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            h = model.get_encoder_output(data)
            features.append(h.cpu().numpy())

    return np.concatenate(features, axis=0)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Preprocess xena browser TCGA data"""
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Omic input file path (TSV format)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Omic output prefix",
        required=True,
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    dataset = OmicsDataset(args.input)

    logger.info(f"Data shape: {dataset.data_array.shape}")
    logger.info(f"Number of samples: {len(dataset)}")
    logger.info(f"Number of features: {dataset.data_array.shape[1]}")

    # Create DataLoader
    batch_size = 32  # As mentioned in the paper
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = dataset.data_array.shape[1]
    hidden_dims = [5000, 128, 64]  # Original architecture from paper

    model = ContractiveAutoEncoder(input_dim, hidden_dims)
    logger.info(
        f"Model architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {input_dim}"
    )

    # Train model
    logger.info("\nTraining Contractive AutoEncoder...")
    train_losses, recon_losses, contract_losses, best_epoch = train_cae(
        model,
        train_loader,
        num_epochs=200,  # As per paper
        learning_rate=0.001,  # As per paper
        lambda_reg=0.0001,  # As per paper
        device=device,
    )

    # Extract features
    logger.info("\nExtracting latent features...")
    features = extract_features(model, train_loader, device)
    logger.info(f"Extracted features shape: {features.shape}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(contract_losses)
    plt.title("Contractive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(f"{args.output}.cae_training_curves.png")
    plt.close()

    logger.info("\nTraining complete! Features extracted and saved.")

    # Save the extracted features
    np.save(f"{args.output}.ef.npy", features)

    # Save the trained model
    torch.save(model.state_dict(), f"{args.output}cae.pth")
