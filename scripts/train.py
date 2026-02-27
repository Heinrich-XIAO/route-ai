import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse

from src.data.dataset import PCBAutoroutingDataset
from src.models.autoencoder import PCBAutoencoder
from src.utils.config import Config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for input_image, output_image in dataloader:
        input_image = input_image.to(device)
        output_image = output_image.to(device)

        optimizer.zero_grad()
        reconstructed = model(input_image)
        loss = criterion(reconstructed, output_image)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_image, output_image in dataloader:
            input_image = input_image.to(device)
            output_image = output_image.to(device)

            reconstructed = model(input_image)
            loss = criterion(reconstructed, output_image)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train PCB Autoencoder")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cuda/mps/cpu/auto)"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    dataset = PCBAutoroutingDataset(
        dataset_name=config.dataset.name,
        split=config.dataset.split,
        image_size=config.dataset.image_size,
        max_samples=config.dataset.max_samples,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    use_pin_memory = device.type in ("cuda", "mps")
    num_workers = 0 if device.type in ("cpu", "mps") else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    model = PCBAutoencoder(
        input_channels=config.model.input_channels,
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.MSELoss()

    checkpoint_dir = Path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(config.training.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.training.grad_clip,
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{config.training.epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
