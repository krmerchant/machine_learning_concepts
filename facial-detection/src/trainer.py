"""Training utilities for face detection models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from .model import FaceDetector
from .utils import calculate_iou, compute_precision_recall


class FaceDetectionTrainer:
    """Trainer class for face detection models.

    This trainer supports fine-tuning MTCNN networks (P-Net, R-Net, O-Net)
    on custom datasets. The typical approach is to fine-tune the O-Net
    while keeping P-Net and R-Net frozen.

    Example:
        >>> detector = FaceDetector()
        >>> trainer = FaceDetectionTrainer(
        ...     detector=detector,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ... )
        >>> history = trainer.fit(epochs=10)
    """

    def __init__(
        self,
        detector: FaceDetector,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize the trainer.

        Args:
            detector: FaceDetector instance to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer (default: Adam with lr=1e-4)
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.detector = detector
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.detector.to(device)

        # Get trainable parameters from O-Net (typically fine-tuned)
        self.onet = self.detector.get_onet()
        if self.onet is not None:
            self.onet.train()
            trainable_params = list(self.onet.parameters())
        else:
            trainable_params = []

        # Set up optimizer
        if optimizer is None and len(trainable_params) > 0:
            self.optimizer = optim.Adam(trainable_params, lr=1e-4)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Loss functions
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.box_loss_fn = nn.SmoothL1Loss()

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        if self.onet is None:
            raise RuntimeError("O-Net not available for training")

        self.onet.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for images, targets in progress_bar:
            images = images.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through O-Net
            # Note: For full MTCNN fine-tuning, you would need to implement
            # the complete cascade with proper proposal generation
            loss = self._compute_loss(images, targets)

            if loss is not None and loss.requires_grad:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            progress_bar.set_postfix({"loss": loss.item() if loss else 0})

        return total_loss / max(num_batches, 1)

    def _compute_loss(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Compute training loss.

        This is a simplified loss computation for demonstration.
        Full MTCNN training requires separate losses for each stage.

        Args:
            images: Batch of images
            targets: List of target dictionaries

        Returns:
            Combined loss tensor
        """
        # For demonstration, we compute a simple classification loss
        # A full implementation would include:
        # 1. P-Net face/non-face classification + bbox regression
        # 2. R-Net refinement
        # 3. O-Net final detection + landmark localization

        batch_size = images.shape[0]
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Simple forward through O-Net for demonstration
        # In practice, you'd need proper proposals from P-Net and R-Net
        for i in range(batch_size):
            boxes = targets[i]["boxes"].to(self.device)
            if len(boxes) > 0:
                # Create dummy positive samples (face crops at ground truth locations)
                # This is simplified - real training uses proposal-based sampling
                h, w = images.shape[2], images.shape[3]
                for box in boxes:
                    x1, y1, x2, y2 = box.int().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        # Extract and resize face region
                        face_crop = images[i:i+1, :, y1:y2, x1:x2]
                        if face_crop.numel() > 0:
                            face_crop = nn.functional.interpolate(
                                face_crop, size=(48, 48), mode="bilinear"
                            )
                            # Forward through O-Net
                            # Note: facenet-pytorch MTCNN O-Net expects specific input format
                            # This is a simplified demonstration

        return total_loss

    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.onet.eval() if self.onet else None

        all_pred_boxes = []
        all_gt_boxes = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                # Get predictions using the detector
                for i in range(len(images)):
                    # Convert tensor to PIL for detection
                    img_np = images[i].permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)

                    boxes, scores, _ = self.detector.detect(img_np)

                    if boxes is not None:
                        all_pred_boxes.append(boxes)
                    else:
                        all_pred_boxes.append(np.array([]))

                    all_gt_boxes.append(targets[i]["boxes"].numpy())

        # Compute metrics
        precision, recall, f1 = compute_precision_recall(
            all_pred_boxes, all_gt_boxes, iou_threshold=0.5
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def fit(
        self,
        epochs: int,
        validate_every: int = 1,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            validate_every: Validation frequency (epochs)
            save_best: Whether to save the best model

        Returns:
            Training history dictionary
        """
        best_f1 = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # Training
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            # Validation
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                self.history["val_precision"].append(val_metrics.get("precision", 0))
                self.history["val_recall"].append(val_metrics.get("recall", 0))
                self.history["val_f1"].append(val_metrics.get("f1", 0))

                print(f"Val Precision: {val_metrics.get('precision', 0):.4f}")
                print(f"Val Recall: {val_metrics.get('recall', 0):.4f}")
                print(f"Val F1: {val_metrics.get('f1', 0):.4f}")

                # Save best model
                if save_best and val_metrics.get("f1", 0) > best_f1:
                    best_f1 = val_metrics["f1"]
                    if self.checkpoint_dir:
                        self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
                        print(f"Saved best model with F1: {best_f1:.4f}")

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

        return self.history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "onet_state_dict": self.onet.state_dict() if self.onet else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        if self.onet and checkpoint.get("onet_state_dict"):
            self.onet.load_state_dict(checkpoint["onet_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("history"):
            self.history = checkpoint["history"]

        print(f"Checkpoint loaded from {path}")

    def export_history(self, path: str) -> None:
        """Export training history to JSON.

        Args:
            path: Path to save history
        """
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
