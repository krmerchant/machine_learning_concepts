"""Face detection model wrapper using MTCNN."""

import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional, Dict, Any
from facenet_pytorch import MTCNN


class FaceDetector:
    """Face detector wrapper around MTCNN.

    Provides a clean interface for face detection with options for
    inference and fine-tuning.

    Example:
        >>> detector = FaceDetector(device='cuda')
        >>> boxes, scores, landmarks = detector.detect(image)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        image_size: int = 160,
        margin: int = 0,
        min_face_size: int = 20,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        keep_all: bool = True,
        select_largest: bool = False,
    ):
        """Initialize the face detector.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            image_size: Output face image size after cropping
            margin: Margin to add around detected face
            min_face_size: Minimum face size to detect
            thresholds: MTCNN confidence thresholds for (P-Net, R-Net, O-Net)
            factor: Scale factor for image pyramid
            keep_all: If True, return all detected faces
            select_largest: If True and keep_all=False, select largest face
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.keep_all = keep_all

        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=factor,
            keep_all=keep_all,
            select_largest=select_largest,
            device=device,
        )

    def detect(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        return_landmarks: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect faces in a single image.

        Args:
            image: Input image (PIL, numpy array, or tensor)
            return_landmarks: Whether to return facial landmarks

        Returns:
            Tuple of (boxes, scores, landmarks) where:
                - boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
                - scores: Array of shape (N,) with confidence scores
                - landmarks: Array of shape (N, 5, 2) with facial landmarks
                  or None if return_landmarks=False
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy().astype(np.uint8))

        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=return_landmarks)

        if boxes is None:
            return None, None, None

        return boxes, probs, landmarks

    def detect_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        return_landmarks: bool = True,
    ) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
        """Detect faces in a batch of images.

        Args:
            images: List of input images
            return_landmarks: Whether to return facial landmarks

        Returns:
            List of (boxes, scores, landmarks) tuples for each image
        """
        results = []
        for image in images:
            boxes, scores, landmarks = self.detect(image, return_landmarks)
            results.append((boxes, scores, landmarks))
        return results

    def extract_faces(
        self,
        image: Union[Image.Image, np.ndarray],
        return_tensors: bool = True,
    ) -> Optional[torch.Tensor]:
        """Extract aligned face crops from an image.

        Args:
            image: Input image
            return_tensors: If True, return tensor; else return numpy array

        Returns:
            Tensor of shape (N, 3, image_size, image_size) with aligned faces
            or None if no faces detected
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        faces = self.mtcnn(image)

        if faces is None:
            return None

        if not return_tensors:
            return faces.cpu().numpy()

        return faces

    def get_pnet(self) -> torch.nn.Module:
        """Get the P-Net (Proposal Network) for fine-tuning.

        Returns:
            P-Net module
        """
        return self.mtcnn.pnet

    def get_rnet(self) -> torch.nn.Module:
        """Get the R-Net (Refine Network) for fine-tuning.

        Returns:
            R-Net module
        """
        return self.mtcnn.rnet

    def get_onet(self) -> torch.nn.Module:
        """Get the O-Net (Output Network) for fine-tuning.

        Returns:
            O-Net module
        """
        return self.mtcnn.onet

    def get_all_networks(self) -> Dict[str, torch.nn.Module]:
        """Get all MTCNN networks.

        Returns:
            Dictionary with 'pnet', 'rnet', 'onet' keys
        """
        return {
            "pnet": self.mtcnn.pnet,
            "rnet": self.mtcnn.rnet,
            "onet": self.mtcnn.onet,
        }

    def set_thresholds(self, thresholds: Tuple[float, float, float]) -> None:
        """Update detection thresholds.

        Args:
            thresholds: New thresholds for (P-Net, R-Net, O-Net)
        """
        self.thresholds = thresholds
        self.mtcnn.thresholds = list(thresholds)

    def set_min_face_size(self, min_face_size: int) -> None:
        """Update minimum face size for detection.

        Args:
            min_face_size: Minimum face size in pixels
        """
        self.min_face_size = min_face_size
        self.mtcnn.min_face_size = min_face_size

    def to(self, device: str) -> "FaceDetector":
        """Move model to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')

        Returns:
            Self for method chaining
        """
        self.device = device
        self.mtcnn.device = torch.device(device)
        if hasattr(self.mtcnn, "pnet") and self.mtcnn.pnet is not None:
            self.mtcnn.pnet = self.mtcnn.pnet.to(device)
        if hasattr(self.mtcnn, "rnet") and self.mtcnn.rnet is not None:
            self.mtcnn.rnet = self.mtcnn.rnet.to(device)
        if hasattr(self.mtcnn, "onet") and self.mtcnn.onet is not None:
            self.mtcnn.onet = self.mtcnn.onet.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"FaceDetector("
            f"device={self.device}, "
            f"min_face_size={self.min_face_size}, "
            f"thresholds={self.thresholds})"
        )
