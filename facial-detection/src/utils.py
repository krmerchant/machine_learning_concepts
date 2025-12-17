"""Utility functions for face detection."""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union, List, Tuple, Optional
from pathlib import Path


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from path.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB format
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculate IoU between two sets of boxes.

    Args:
        boxes1: Array of shape (N, 4) with boxes [x1, y1, x2, y2]
        boxes2: Array of shape (M, 4) with boxes [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M)
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]

    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = calculate_iou(boxes1[i], boxes2[j])

    return iou_matrix


def draw_boxes(
    image: Union[Image.Image, np.ndarray],
    boxes: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    landmarks: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 10),
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw bounding boxes and landmarks on an image.

    Args:
        image: Input image (PIL Image or numpy array)
        boxes: Predicted bounding boxes of shape (N, 4) as [x1, y1, x2, y2]
        scores: Confidence scores of shape (N,)
        landmarks: Facial landmarks of shape (N, 5, 2) or (N, 10)
        gt_boxes: Ground truth boxes of shape (M, 4) for comparison
        figsize: Figure size
        show: Whether to display the figure
        save_path: Path to save the figure

    Returns:
        matplotlib Figure object
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    # Draw ground truth boxes in green
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = box[:4]
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor="green", facecolor="none",
                linestyle="--", label="Ground Truth"
            )
            ax.add_patch(rect)

    # Draw predicted boxes in red
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            width = x2 - x1
            height = y2 - y1

            color = "red"
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Add score label
            if scores is not None:
                label = f"{scores[i]:.2f}"
                ax.text(
                    x1, y1 - 5, label,
                    color="white", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.8)
                )

    # Draw landmarks
    if landmarks is not None:
        for lm in landmarks:
            if lm is not None:
                # Handle different landmark formats
                if lm.ndim == 1 and len(lm) == 10:
                    # Flat format: [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
                    points = lm.reshape(2, 5).T
                elif lm.ndim == 2 and lm.shape == (5, 2):
                    points = lm
                else:
                    continue

                ax.scatter(points[:, 0], points[:, 1], c="cyan", s=20, marker="o")

    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig


def compute_precision_recall(
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute precision and recall for face detection.

    Args:
        pred_boxes: List of predicted boxes per image
        gt_boxes: List of ground truth boxes per image
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, gts in zip(pred_boxes, gt_boxes):
        if len(preds) == 0:
            total_fn += len(gts)
            continue

        if len(gts) == 0:
            total_fp += len(preds)
            continue

        iou_matrix = calculate_batch_iou(preds, gts)

        matched_gt = set()
        for i in range(len(preds)):
            max_iou_idx = np.argmax(iou_matrix[i])
            max_iou = iou_matrix[i, max_iou_idx]

            if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
                total_tp += 1
                matched_gt.add(max_iou_idx)
            else:
                total_fp += 1

        total_fn += len(gts) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
