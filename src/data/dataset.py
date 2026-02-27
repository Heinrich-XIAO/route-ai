from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np


class PCBAutoroutingDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "tscircuit/zero-obstacle-high-density-z01",
        split: str = "train",
        image_size: int = 256,
        max_samples: int | None = None,
        transform=None,
    ):
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_size = image_size
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self) -> int:
        if self.max_samples is not None:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]

        cond_image = item["cond_image"].convert("RGB")
        output_image = item["output_image"].convert("RGB")

        cond_image = cond_image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        output_image = output_image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

        cond_array = np.array(cond_image, dtype=np.float32) / 255.0
        output_array = np.array(output_image, dtype=np.float32) / 255.0

        cond_tensor = torch.from_numpy(cond_array).permute(2, 0, 1)
        output_tensor = torch.from_numpy(output_array).permute(2, 0, 1)

        if self.transform:
            cond_tensor = self.transform(cond_tensor)
            output_tensor = self.transform(output_tensor)

        return cond_tensor, output_tensor
