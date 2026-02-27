#!/usr/bin/env python3
"""Test script for PCB Autoencoder - run inference on a sample."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from PIL import Image
import numpy as np

from src.models.autoencoder import PCBAutoencoder
from src.data.dataset import PCBAutoroutingDataset


def test_model(checkpoint_path: str | None = None, sample_idx: int = 0):
    """Test the model on a sample from the dataset."""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model = PCBAutoencoder(
        input_channels=3,
        latent_dim=256,
        base_channels=64,
    ).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found, using random weights")
    
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = PCBAutoroutingDataset(max_samples=10)
    
    # Get sample
    input_image, output_image = dataset[sample_idx]
    
    # Add batch dimension and move to device
    input_image = input_image.unsqueeze(0).to(device)
    output_image = output_image.unsqueeze(0).to(device)
    
    # Run inference
    print(f"Running inference on sample {sample_idx}...")
    with torch.no_grad():
        reconstructed = model(input_image, output_image)
    
    # Calculate loss
    criterion = torch.nn.MSELoss()
    loss = criterion(reconstructed, output_image)
    print(f"Reconstruction loss: {loss.item():.6f}")
    
    # Save images
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Convert tensors to images
    def tensor_to_image(tensor):
        img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)
    
    input_img = tensor_to_image(input_image)
    output_img = tensor_to_image(output_image)
    recon_img = tensor_to_image(reconstructed)
    
    input_img.save(output_dir / f"sample_{sample_idx}_input.png")
    output_img.save(output_dir / f"sample_{sample_idx}_target.png")
    recon_img.save(output_dir / f"sample_{sample_idx}_reconstructed.png")
    
    print(f"Saved images to {output_dir}/")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Test PCB Autoencoder")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Sample index to test"
    )
    args = parser.parse_args()
    
    test_model(args.checkpoint, args.sample)


if __name__ == "__main__":
    main()
