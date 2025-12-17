"""WIDER FACE dataset loader for face detection."""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class WIDERFaceDataset(Dataset):
    """PyTorch Dataset for WIDER FACE face detection benchmark.

    The WIDER FACE dataset contains 32,203 images with 393,703 labeled faces.
    Download from: http://shuoyang1213.me/WIDERFACE/

    Expected directory structure:
        root/
        ├── WIDER_train/
        │   └── images/
        │       ├── 0--Parade/
        │       │   ├── 0_Parade_marchingband_1_5.jpg
        │       │   └── ...
        │       └── ...
        ├── WIDER_val/
        │   └── images/
        │       └── ...
        └── wider_face_split/
            ├── wider_face_train_bbx_gt.txt
            └── wider_face_val_bbx_gt.txt

    Example:
        >>> dataset = WIDERFaceDataset(root='./data', split='train')
        >>> image, target = dataset[0]
        >>> boxes = target['boxes']  # Shape: (N, 4)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_size: int = 10,
        augment: bool = False,
    ):
        """Initialize WIDER FACE dataset.

        Args:
            root: Root directory containing WIDER FACE data
            split: Dataset split ('train' or 'val')
            transform: Optional transform to apply to images
            min_size: Minimum face size to include (filters tiny faces)
            augment: Whether to apply data augmentation
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.min_size = min_size
        self.augment = augment

        # Set paths based on split
        if split == "train":
            self.image_dir = self.root / "WIDER_train" / "images"
            self.annotation_file = self.root / "wider_face_split" / "wider_face_train_bbx_gt.txt"
        elif split == "val":
            self.image_dir = self.root / "WIDER_val" / "images"
            self.annotation_file = self.root / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'val'.")

        # Load annotations
        self.samples = self._load_annotations()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Parse WIDER FACE annotation file.

        Annotation format:
            image_path
            num_faces
            x1 y1 w h blur expression illumination invalid occlusion pose
            ...

        Returns:
            List of dictionaries with 'image_path' and 'boxes' keys
        """
        samples = []

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}\n"
                f"Please download WIDER FACE from http://shuoyang1213.me/WIDERFACE/"
            )

        with open(self.annotation_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            # Read image path
            image_path = lines[i].strip()
            i += 1

            # Read number of faces
            num_faces = int(lines[i].strip())
            i += 1

            boxes = []

            # Handle case where num_faces is 0
            if num_faces == 0:
                # Skip the placeholder line
                i += 1
            else:
                # Read face annotations
                for _ in range(num_faces):
                    parts = lines[i].strip().split()
                    i += 1

                    # Parse bounding box (x, y, width, height)
                    x, y, w, h = map(int, parts[:4])

                    # Filter invalid or tiny faces
                    invalid = int(parts[7]) if len(parts) > 7 else 0
                    if invalid or w < self.min_size or h < self.min_size:
                        continue

                    # Convert to [x1, y1, x2, y2] format
                    boxes.append([x, y, x + w, y + h])

            # Only add samples with valid faces
            if len(boxes) > 0:
                samples.append({
                    "image_path": str(self.image_dir / image_path),
                    "boxes": np.array(boxes, dtype=np.float32),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, target_dict) where:
                - image_tensor: Shape (3, H, W)
                - target_dict: {'boxes': tensor (N, 4), 'labels': tensor (N,)}
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        boxes = sample["boxes"].copy()

        # Apply augmentation
        if self.augment:
            image, boxes = self._augment(image, boxes)

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)

        # Create target dictionary
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),  # All faces have label 1
        }

        return image, target

    def _augment(
        self,
        image: Image.Image,
        boxes: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        """Apply data augmentation.

        Args:
            image: Input image
            boxes: Bounding boxes of shape (N, 4)

        Returns:
            Augmented image and boxes
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            w = image.width
            boxes_flipped = boxes.copy()
            boxes_flipped[:, 0] = w - boxes[:, 2]  # x1 = w - x2
            boxes_flipped[:, 2] = w - boxes[:, 0]  # x2 = w - x1
            boxes = boxes_flipped

        # Random color jitter
        if np.random.random() > 0.5:
            image = TF.adjust_brightness(image, np.random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, np.random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, np.random.uniform(0.8, 1.2))

        return image, boxes

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata about an image without loading it.

        Args:
            idx: Sample index

        Returns:
            Dictionary with image path and number of faces
        """
        sample = self.samples[idx]
        return {
            "path": sample["image_path"],
            "num_faces": len(sample["boxes"]),
        }

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """Custom collate function for DataLoader.

        Since images may have different numbers of faces, we can't stack
        targets into a single tensor.

        Args:
            batch: List of (image, target) tuples

        Returns:
            Tuple of (stacked_images, list_of_targets)
        """
        images = []
        targets = []

        for image, target in batch:
            images.append(image)
            targets.append(target)

        # Stack images (assumes same size after transforms)
        images = torch.stack(images, dim=0)

        return images, targets


def create_wider_face_transforms(
    train: bool = True,
    image_size: Tuple[int, int] = (640, 640),
) -> T.Compose:
    """Create standard transforms for WIDER FACE.

    Args:
        train: Whether these are training transforms
        image_size: Target image size (H, W)

    Returns:
        Composed transforms
    """
    if train:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
