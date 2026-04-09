"""Perception engine — camera frames, VLM, depth, spatial reasoning."""

from roboclaw.embodied.perception.camera_grabber import (
    camera_configs,
    frame_to_base64,
    frame_to_bytes,
    frame_to_pil,
    grab_all_frames,
    grab_frame,
    save_frame,
)
from roboclaw.embodied.perception.depth import DepthEstimator
from roboclaw.embodied.perception.spatial import SpatialQuery, SpatialReasoner
from roboclaw.embodied.perception.vlm import SceneDescription, VLM

__all__ = [
    "VLM",
    "SceneDescription",
    "DepthEstimator",
    "SpatialQuery",
    "SpatialReasoner",
    "grab_frame",
    "grab_all_frames",
    "camera_configs",
    "frame_to_base64",
    "frame_to_bytes",
    "frame_to_pil",
    "save_frame",
]
