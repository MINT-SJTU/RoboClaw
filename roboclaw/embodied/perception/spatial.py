"""Spatial reasoning combining depth and VLM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roboclaw.embodied.perception.camera_grabber import grab_frame
from roboclaw.embodied.perception.depth import DepthEstimator


@dataclass
class SpatialQuery:
    """Structured spatial relationship result."""
    object_a: str
    object_b: str
    relation: str  # "left_of", "right_of", "above", "below", "in_front", "behind"
    confidence: float


class SpatialReasoner:
    """
    Lightweight spatial reasoning using depth map statistics.

    Uses DA3 depth to determine which object is closer/farther and
    projects bounding boxes to infer left/right/above/below relationships.
    """

    def __init__(self):
        self._depth = DepthEstimator()

    def relation_between(
        self,
        camera_alias: str,
        object_a: str,
        object_b: str,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> SpatialQuery:
        """
        Infer spatial relationship between two objects.

        Args:
            camera_alias: Camera to use.
            object_a, object_b: Object names (for labeling output).
            bbox_a, bbox_b: (x_min, y_min, x_max, y_max) in pixel coords.

        Returns:
            SpatialQuery with relation and confidence estimate.
        """
        frame = grab_frame(camera_alias)
        if frame is None:
            return SpatialQuery(object_a, object_b, "unknown", 0.0)

        depth_map = self._depth.estimate(frame)
        if depth_map is None:
            return self._relation_from_centers(object_a, object_b, bbox_a, bbox_b)

        # Average depth of each object's bounding box
        d_a = self._mean_depth(depth_map, bbox_a)
        d_b = self._mean_depth(depth_map, bbox_b)

        # Horizontal and vertical center positions
        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2

        relations: list[tuple[str, float]] = []

        # Z-axis (depth)
        if d_a is not None and d_b is not None:
            if d_a < d_b:
                relations.append(("a_is_closer", abs(d_a - d_b) / max(d_b, 1e-6)))
            else:
                relations.append(("b_is_closer", abs(d_a - d_b) / max(d_a, 1e-6)))

        # X-axis (left/right)
        dx = cx_a - cx_b
        if abs(dx) > 5:
            if dx < 0:
                relations.append(("a_left_of_b", min(abs(dx) / max(frame.shape[1], 1), 1.0)))
            else:
                relations.append(("a_right_of_b", min(abs(dx) / max(frame.shape[1], 1), 1.0)))

        # Y-axis (above/below)
        dy = cy_a - cy_b
        if abs(dy) > 5:
            if dy < 0:
                relations.append(("a_above_b", min(abs(dy) / max(frame.shape[0], 1), 1.0)))
            else:
                relations.append(("a_below_b", min(abs(dy) / max(frame.shape[0], 1), 1.0)))

        if not relations:
            return SpatialQuery(object_a, object_b, "overlapping", 0.5)

        best_rel, best_conf = max(relations, key=lambda x: x[1])
        return SpatialQuery(object_a, object_b, best_rel, best_conf)

    def _mean_depth(self, depth: np.ndarray, bbox: tuple[int, int, int, int]) -> float | None:
        x0, y0, x1, y1 = bbox
        x0, x1 = max(0, x0), min(depth.shape[1], x1)
        y0, y1 = max(0, y0), min(depth.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return float(depth[y0:y1, x0:x1].mean())

    def _relation_from_centers(
        self,
        a: str, b: str,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> SpatialQuery:
        """Fallback using center positions without depth."""
        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2
        if abs(cx_a - cx_b) >= abs(cy_a - cy_b):
            rel = "a_left_of_b" if cx_a < cx_b else "a_right_of_b"
        else:
            rel = "a_above_b" if cy_a < cy_b else "a_below_b"
        return SpatialQuery(a, b, rel, 0.6)
