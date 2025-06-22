import os
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import argparse
import logging

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class OmicsDataset(Dataset):
    """Dataset class for omics data"""

    def __init__(
        self, data_path, transform=None, normalization="minmax", scaler_path=None
    ):
        # Read the TSV file
        self.data = pd.read_csv(data_path, sep="\t", index_col=0)

        # Transpose to have samples as rows and features as columns
        self.data = self.data.T

        # Convert to numpy array
        self.data_array = self.data.values.astype(np.float32)

        # Apply normalization
        if normalization == "minmax":
            self.scaler = MinMaxScaler()
        elif normalization == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        if self.scaler:
            self.data_array = self.scaler.fit_transform(self.data_array)

        if scaler_path:
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            print(f"Saved scaler to {scaler_path}")

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
    Improved Contractive AutoEncoder with better initialization and architecture options
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[5000, 128, 64],
        dropout_rate=0.2,
        activation="relu",
        use_batch_norm=False,
        init_method="xavier",
    ):
        super(ContractiveAutoEncoder, self).__init__()

        # Store dimensions for later use
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)  # Helps prevent dying neurons
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(self.hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, h_dim, bias=True)

            # Initialize weights
            if init_method == "xavier":
                nn.init.xavier_uniform_(linear.weight)
            elif init_method == "he":
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(linear.weight)

            # Initialize bias to small positive values to help prevent dying ReLU
            nn.init.constant_(linear.bias, 0.01)

            encoder_layers.append(linear)

            # Batch normalization (optional)
            if use_batch_norm and i < len(self.hidden_dims) - 1:
                encoder_layers.append(nn.BatchNorm1d(h_dim))

            # Activation
            encoder_layers.append(self.activation)

            # Dropout (not on bottleneck)
            if i < len(self.hidden_dims) - 1:
                encoder_layers.append(nn.Dropout(dropout_rate))

            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        reversed_dims = self.hidden_dims[::-1]

        for i in range(len(reversed_dims) - 1):
            linear = nn.Linear(reversed_dims[i], reversed_dims[i + 1], bias=True)

            # Initialize weights
            if init_method == "xavier":
                nn.init.xavier_uniform_(linear.weight)
            elif init_method == "he":
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(linear.weight)

            nn.init.constant_(linear.bias, 0.01)

            decoder_layers.append(linear)

            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(reversed_dims[i + 1]))

            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout_rate))

        # Final decoder layer with sigmoid activation
        final_linear = nn.Linear(reversed_dims[-1], input_dim, bias=True)
        nn.init.xavier_uniform_(final_linear.weight)
        decoder_layers.append(final_linear)
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

        # Store the first encoder layer for contractive loss
        self.encoder_weight = None
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                self.encoder_weight = module.weight
                break

    def forward(self, x):
        # Get hidden representation
        h = self.encoder(x)
        # Get reconstruction
        x_recon = self.decoder(h)
        return h, x_recon

    def get_encoder_output(self, x):
        """Get only the encoder output (latent representation)"""
        return self.encoder(x)


def analyze_hidden_activations(model, data_loader, device="cuda"):
    """
    Analyze the hidden layer activations to detect dead neurons
    """
    model.eval()
    model = model.to(device)

    all_hidden = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            h = model.get_encoder_output(data)
            all_hidden.append(h.cpu().numpy())

    all_hidden = np.concatenate(all_hidden, axis=0)

    # Calculate statistics
    mean_activation = np.mean(all_hidden, axis=0)
    std_activation = np.std(all_hidden, axis=0)
    zero_fraction = np.mean(all_hidden == 0, axis=0)

    # Count dead neurons (always zero)
    dead_neurons = np.sum(zero_fraction == 1.0)
    nearly_dead = np.sum(zero_fraction > 0.95)

    logger.info("\n" + "=" * 50)
    logger.info("HIDDEN LAYER ANALYSIS")
    logger.info("=" * 50)
    logger.info(f"Hidden dimension: {all_hidden.shape[1]}")
    logger.info(
        f"Dead neurons (always 0): {dead_neurons} ({dead_neurons/all_hidden.shape[1]*100:.1f}%)"
    )
    logger.info(
        f"Nearly dead (>95% zeros): {nearly_dead} ({nearly_dead/all_hidden.shape[1]*100:.1f}%)"
    )
    logger.info(f"Mean activation: {np.mean(mean_activation):.4f}")
    logger.info(f"Std activation: {np.mean(std_activation):.4f}")

    # Plot activation distribution
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(mean_activation, bins=50, alpha=0.7)
    plt.xlabel("Mean Activation")
    plt.ylabel("Count")
    plt.title("Distribution of Mean Activations")

    plt.subplot(1, 3, 2)
    plt.hist(zero_fraction, bins=50, alpha=0.7)
    plt.xlabel("Fraction of Zeros")
    plt.ylabel("Count")
    plt.title("Distribution of Zero Fractions")

    plt.subplot(1, 3, 3)
    plt.scatter(range(len(mean_activation)), sorted(mean_activation), s=1)
    plt.xlabel("Neuron Index (sorted)")
    plt.ylabel("Mean Activation")
    plt.title("Sorted Mean Activations")

    plt.tight_layout()
    plt.savefig("hidden_layer_analysis.png")
    plt.close()

    return {
        "dead_neurons": dead_neurons,
        "nearly_dead": nearly_dead,
        "mean_activation": np.mean(mean_activation),
        "std_activation": np.mean(std_activation),
    }


def adaptive_lambda_schedule(epoch, initial_lambda=0.0001, warmup_epochs=20):
    """
    Adaptive lambda schedule: start with lower lambda to allow learning
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lambda * (epoch / warmup_epochs)
    else:
        return initial_lambda


def train_cae(
    model,
    train_loader,
    num_epochs=200,
    learning_rate=0.001,
    lambda_reg=0.00001,
    device="cuda",
    patience=20,
    use_warmup=True,
    monitor_interval=20,
):
    """
    Training with monitoring and adaptive regularization
    """
    model = model.to(device)

    # Use different learning rates for encoder and decoder
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    # Decoder can have slightly higher learning rate
    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": learning_rate},
            {"params": decoder_params, "lr": learning_rate * 1.5},
        ]
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

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

        # Adaptive lambda
        if use_warmup:
            current_lambda = adaptive_lambda_schedule(epoch, lambda_reg)
        else:
            current_lambda = lambda_reg

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()

            # Forward pass
            h, x_recon = model(data)

            # Calculate loss with current lambda
            loss, recon_loss, contract_loss = contractive_loss(
                data, x_recon, h, model.encoder_weight, current_lambda
            )

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Learning rate scheduling
        scheduler.step(avg_loss)

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
                f"Contract Loss: {avg_contract_loss:.4f}, "
                f"Lambda: {current_lambda:.6f}"
            )

            if best_model_state is not None:
                logger.info(f"  Best Loss: {best_loss:.4f} at epoch {best_epoch+1}")

        # Monitor hidden layer health
        if (epoch + 1) % monitor_interval == 0:
            stats = analyze_hidden_activations(model, train_loader, device)
            if (
                stats["dead_neurons"] > model.hidden_dims[-1] * 0.5
            ):  # More than 50% dead
                logger.warning(
                    f"WARNING: {stats['dead_neurons']} dead neurons detected!"
                )

        # Early stopping
        # if epochs_no_improve >= patience:
        #     logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
        #     logger.info(
        #         f"Best model was at epoch {best_epoch+1} with loss {best_loss:.4f}"
        #     )
        #     break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored best model from epoch {best_epoch+1}")

    return train_losses, recon_losses, contract_losses, best_epoch


# Keep the original contractive_loss function as is
def contractive_loss(x, x_recon, h, W, lambda_reg=1e-4):
    """
    Compute the Contractive AutoEncoder Loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    # Get the output of first linear layer
    h_linear = (
        torch.matmul(x, W.t()) + W.bias
        if hasattr(W, "bias")
        else torch.matmul(x, W.t())
    )

    # Derivative of ReLU
    dh = (h_linear > 0).float()

    # Efficient computation as suggested in the paper
    w_squared = W**2
    contract_loss = torch.sum(
        torch.mm(dh**2, w_squared.sum(dim=1).unsqueeze(1))
    ) / x.size(0)

    # Total loss
    total_loss = recon_loss + lambda_reg * contract_loss

    return total_loss, recon_loss, contract_loss


def extract_features(model, data_loader, device="cuda"):
    """
    Extract latent features from trained model
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
        description="""Train Contractive AutoEncoder on omics data"""
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
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "leaky_relu", "elu"],
        default="leaky_relu",
        help="Activation function (default: leaky_relu to prevent dying neurons)",
    )
    parser.add_argument(
        "--batch-norm", action="store_true", help="Use batch normalization"
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.0001,
        help="Regularization coefficient (default: 0.0001)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    scaler_path = f"{args.output}_scaler.pkl"
    dataset = OmicsDataset(args.input, normalization="minmax", scaler_path=scaler_path)

    logger.info(f"Data shape: {dataset.data_array.shape}")
    logger.info(f"Number of samples: {len(dataset)}")
    logger.info(f"Number of features: {dataset.data_array.shape[1]}")

    # Create DataLoader
    batch_size = 32  # As mentioned in the paper
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with improvements
    input_dim = dataset.data_array.shape[1]
    hidden_dims = [5000, 128, 64]  # Original architecture from paper

    model = ContractiveAutoEncoder(
        input_dim,
        hidden_dims,
        activation=args.activation,
        use_batch_norm=args.batch_norm,
        init_method="xavier_normal",
    )

    logger.info(
        f"Model architecture: {input_dim} -> {' -> '.join(map(str, model.hidden_dims))} -> {input_dim}"
    )

    # Train model with improvements
    logger.info("\nTraining Improved Contractive AutoEncoder...")
    train_losses, recon_losses, contract_losses, best_epoch = train_cae(
        model,
        train_loader,
        num_epochs=200,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg,
        device=device,
        use_warmup=True,
    )

    # Final analysis of hidden layer
    logger.info("\nFinal hidden layer analysis:")
    final_stats = analyze_hidden_activations(model, train_loader, device)

    # Extract features
    logger.info("\nExtracting latent features...")
    features = extract_features(model, train_loader, device)
    logger.info(f"Extracted features shape: {features.shape}")

    # Check for all-zero features
    zero_features = np.sum(np.all(features == 0, axis=0))
    logger.info(f"Features that are all zeros: {zero_features}/{features.shape[1]}")

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
    plt.savefig(f"{args.output}_cae_training_curves.png")
    plt.close()

    logger.info("\nTraining complete! Features extracted and saved.")

    # Save the extracted features
    np.save(f"{args.output}_ef.npy", features)

    # Save the trained model
    torch.save(model.state_dict(), f"{args.output}_cae_model.pth")

    # Save model metadata
    metadata = {
        "input_dim": input_dim,
        "hidden_dims": model.hidden_dims,
        "activation": args.activation,
        "batch_norm": args.batch_norm,
        "best_epoch": best_epoch,
        "final_stats": final_stats,
    }
    torch.save(metadata, f"{args.output}_metadata.pth")
