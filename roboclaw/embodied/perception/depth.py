"""Depth estimation using Depth Anything 3 (via HuggingFace transformers)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DepthEstimator:
    """
    Single-frame monocular depth estimation via Depth Anything 3.

    Falls back gracefully when the model is not available, returning None
    instead of crashing so the rest of the perception stack stays functional.
    """

    _instance: "DepthEstimator | None" = None
    _pipeline: Any = None  # set after first successful load

    def __new__(cls) -> "DepthEstimator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            import torch
        except ImportError:
            return

        try:
            self._processor = AutoImageProcessor.from_pretrained(
                "Depth-Anything/Depth-Anything-V2-Base-hf",
                trust_remote_code=True,
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                "Depth-Anything/Depth-Anything-V2-Base-hf",
                trust_remote_code=True,
            )
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
        except Exception:
            self._processor = None
            self._model = None

    def estimate(self, frame: np.ndarray) -> NDArray[np.float32] | None:
        """
        Estimate depth map from a BGR frame.

        Args:
            frame: HWC BGR numpy array from OpenCV.

        Returns:
            HxW float32 depth map (normalized to roughly [0, 1]),
            or None if the model is not available.
        """
        if self._model is None:
            return None

        from PIL import Image
        import torch

        rgb = np.ascontiguousarray(frame[..., ::-1])  # BGR → RGB
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            inputs = self._processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Normalize to [0, 1] for downstream consistency
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)

        return depth.astype(np.float32)

    def depth_at_point(self, depth_map: NDArray[np.float32], u: int, v: int) -> float | None:
        """Read depth value at pixel (u=col, v=row). Returns None if out of bounds."""
        if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
            return None
        return float(depth_map[v, u])

    def pixel_to_3d(
        self,
        depth_map: NDArray[np.float32],
        u: int,
        v: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scale: float = 1.0,
    ) -> tuple[float, float, float] | None:
        """
        Back-project a pixel to 3D camera coordinates.

        Args:
            depth_map: HxW depth map (normalized [0,1]).
            u, v: Pixel column and row.
            fx, fy: Camera focal lengths in pixels.
            cx, cy: Camera principal point.
            scale: Depth map scale factor (multiply normalized depth to get meters).

        Returns:
            (x, y, z) in camera frame, or None if out of bounds.
        """
        d = self.depth_at_point(depth_map, u, v)
        if d is None:
            return None
        z = d * scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return (float(x), float(y), float(z))
