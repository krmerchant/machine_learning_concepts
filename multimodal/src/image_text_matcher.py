"""CLIP-based image-text matching."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor


class ImageTextMatcher:
    """Image-text similarity matcher using CLIP.

    Wraps a HuggingFace CLIP model to compute similarity between
    images and text descriptions.

    Example:
        >>> matcher = ImageTextMatcher()
        >>> matcher.add_texts(["a dog", "a cat", "a car"])
        >>> matcher.add_images(image_tensor)  # (N, 3, H, W)
        >>> similarity = matcher.compute_similarity()  # (N, 3) matrix
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """Initialize the image-text matcher.

        Args:
            model_name: HuggingFace model identifier for CLIP
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Storage for embeddings
        self._text_embeddings: Optional[torch.Tensor] = None
        self._image_embeddings: Optional[torch.Tensor] = None
        self._texts: List[str] = []

    def add_texts(self, texts: List[str]) -> None:
        """Add text descriptions and compute their embeddings.

        Args:
            texts: List of text descriptions
        """
        self._texts = texts

        # Process texts
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Compute text embeddings
        with torch.no_grad():
            text_outputs = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # Normalize embeddings
            self._text_embeddings = F.normalize(text_outputs, p=2, dim=-1)

    def add_images(
        self,
        images: Union[torch.Tensor, List[Image.Image], List[np.ndarray]],
    ) -> None:
        """Add images and compute their embeddings.

        Args:
            images: Images in one of the following formats:
                - torch.Tensor of shape (N, 3, H, W) with values in [0, 1]
                - List of PIL Images
                - List of numpy arrays (H, W, 3) with values in [0, 255]
        """
        # Convert torch tensor to list of PIL images
        if isinstance(images, torch.Tensor):
            images = self._tensor_to_pil_list(images)
        elif isinstance(images, np.ndarray):
            images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        elif isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], np.ndarray):
                images = [Image.fromarray(img.astype(np.uint8)) for img in images]

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Compute image embeddings
        with torch.no_grad():
            image_outputs = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            # Normalize embeddings
            self._image_embeddings = F.normalize(image_outputs, p=2, dim=-1)

    def compute_similarity(self) -> torch.Tensor:
        """Compute similarity matrix between images and texts.

        Returns:
            Similarity matrix of shape (num_images, num_texts)
            Values are cosine similarities in range [-1, 1]

        Raises:
            ValueError: If no images or texts have been added
        """
        if self._image_embeddings is None:
            raise ValueError("No images added. Call add_images() first.")
        if self._text_embeddings is None:
            raise ValueError("No texts added. Call add_texts() first.")

        # Compute cosine similarity
        # image_embeddings: (N, D), text_embeddings: (M, D)
        # Result: (N, M)
        similarity = torch.matmul(
            self._image_embeddings, self._text_embeddings.T
        )

        return similarity

    def compute_softmax_scores(
        self,
        temperature: float = 100.0,
    ) -> torch.Tensor:
        """Compute softmax probabilities over texts for each image.

        Applies softmax over text descriptions, converting cosine similarities
        to probabilities that sum to 1. Useful for ranking/verification where
        you want to know which text best matches each image.

        Args:
            temperature: Scaling factor before softmax (CLIP default=100).
                Higher = sharper distribution, lower = more uniform.

        Returns:
            Probability matrix of shape (num_images, num_texts)
            Each row sums to 1.0

        Example:
            >>> matcher.add_texts(["a dog", "a cat", "background"])
            >>> matcher.add_images(crops)
            >>> probs = matcher.compute_softmax_scores()
            >>> # probs[i] = probability distribution over texts for image i
            >>> # probs[i].argmax() = index of best matching text
        """
        similarity = self.compute_similarity()

        # Apply temperature scaling and softmax over text dimension
        # Shape: (num_images, num_texts)
        probs = F.softmax(similarity * temperature, dim=1)

        return probs

    def verify_detections(
        self,
        positive_idx: int = 0,
        min_score: Optional[float] = None,
    ) -> torch.Tensor:
        """Verify which images match the positive text description.

        Uses softmax ranking to determine if the positive text wins
        against other (negative) text descriptions.

        Args:
            positive_idx: Index of the positive text in the texts list
                (typically 0 if you add positive text first)
            min_score: Optional minimum cosine similarity threshold.
                If set, also requires raw similarity > min_score.

        Returns:
            Boolean tensor of shape (num_images,) indicating verified detections

        Example:
            >>> matcher.add_texts(["person in red", "person in blue", "background"])
            >>> matcher.add_images(detection_crops)
            >>> verified = matcher.verify_detections(positive_idx=0)
            >>> # verified[i] = True if image i best matches "person in red"
        """
        similarity = self.compute_similarity()

        # Check if positive text wins (has highest score)
        best_match_idx = similarity.argmax(dim=1)
        verified = best_match_idx == positive_idx

        # Optionally apply minimum score threshold
        if min_score is not None:
            above_threshold = similarity[:, positive_idx] > min_score
            verified = verified & above_threshold

        return verified

    def get_best_text_match(self, image_idx: int = 0) -> tuple:
        """Get the best matching text for an image.

        Args:
            image_idx: Index of the image

        Returns:
            Tuple of (text, similarity_score)
        """
        similarity = self.compute_similarity()
        scores = similarity[image_idx]
        best_idx = scores.argmax().item()
        return self._texts[best_idx], scores[best_idx].item()

    def get_best_image_match(self, text_idx: int = 0) -> tuple:
        """Get the best matching image index for a text.

        Args:
            text_idx: Index of the text

        Returns:
            Tuple of (image_index, similarity_score)
        """
        similarity = self.compute_similarity()
        scores = similarity[:, text_idx]
        best_idx = scores.argmax().item()
        return best_idx, scores[best_idx].item()

    def clear(self) -> None:
        """Clear all stored texts and images."""
        self._text_embeddings = None
        self._image_embeddings = None
        self._texts = []

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert a tensor batch to list of PIL images.

        Args:
            tensor: Tensor of shape (N, 3, H, W) with values in [0, 1]

        Returns:
            List of PIL Images
        """
        images = []
        for i in range(tensor.shape[0]):
            img = tensor[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(img))
        return images

    @property
    def num_texts(self) -> int:
        """Number of texts currently stored."""
        return len(self._texts)

    @property
    def num_images(self) -> int:
        """Number of images currently stored."""
        if self._image_embeddings is None:
            return 0
        return self._image_embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        """Dimension of the CLIP embedding space."""
        return self.model.config.projection_dim

    def __repr__(self) -> str:
        return (
            f"ImageTextMatcher("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"num_images={self.num_images}, "
            f"num_texts={self.num_texts})"
        )
