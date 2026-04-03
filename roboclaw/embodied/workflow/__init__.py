from __future__ import annotations

# -- features (scalars, row resolution, series, joints) --
from .features import (
    ACTION_FIELD_CANDIDATES,
    FRAME_INDEX_FIELD_CANDIDATES,
    STATE_FIELD_CANDIDATES,
    TASK_FIELD_CANDIDATES,
    TIMESTAMP_FIELD_CANDIDATES,
    build_episode_feature_vector,
    build_episode_sequence,
    build_joint_trajectory_payload,
    clamp,
    coerce_vector,
    extract_joint_names,
    first_present_value,
    mean,
    normalize_joint_names,
    normalize_scalar_series,
    percentile,
    resolve_action_vector,
    resolve_frame_index,
    resolve_state_vector,
    resolve_task_value,
    resolve_timestamp,
    sample_indices,
    sample_sequence,
    stdev,
    summarize_series,
)

# -- DTW distance algorithms --
from .dtw import (
    CARTESIAN_20D_GROUP_WEIGHTS,
    CARTESIAN_20D_WINDOW_RATIO,
    DEFAULT_DTW_HUBER_DELTA,
    average_vectors,
    build_distance_matrix,
    build_distance_matrix_with_progress,
    dtw_alignment,
    dtw_distance,
    euclidean_distance,
    grouped_huber_distance,
    huber_loss,
    resolve_dtw_configuration,
    vector_distance,
)

# -- clustering (k-medoids, DBA) --
from .clustering import (
    compute_dba_barycenter,
    discover_prototype_clusters,
    refine_clusters_with_dba,
)

# -- canonical trajectory --
from .canonical import (
    ALOHA_ARM_JOINT_ORDER,
    CANONICAL_GROUP_SLICES,
    CanonicalTrajectory,
    build_canonical_trajectory,
    build_cartesian_canonical_trajectory,
    build_cartesian_feature_rows,
    build_joint_canonical_trajectory,
)

# -- propagation (quality tags, annotations, grasp/place) --
from .propagation import (
    build_confidence_payload,
    build_hf_annotation_rows,
    build_phase_progress,
    derive_quality_tags,
    detect_grasp_place_events,
    propagate_annotation_spans,
)

# -- validators --
from .validators import (
    VALIDATOR_REGISTRY,
    finalize_validator,
    load_episode_data,
    make_issue,
    run_quality_validators,
    validate_action,
    validate_depth_assets,
    validate_metadata,
    validate_timing,
    validate_visual_assets,
)

# -- workflow state persistence --
from .state import (
    init_workflow_state,
    load_annotations,
    load_propagation_results,
    load_prototype_results,
    load_quality_results,
    load_workflow_state,
    save_annotations,
    save_propagation_results,
    save_prototype_results,
    save_quality_results,
    save_workflow_state,
)

from .exports import (
    build_quality_result_rows,
    build_text_annotation_rows,
    dataset_quality_parquet_path,
    dataset_text_annotations_parquet_path,
    export_quality_csv,
    publish_quality_metadata_parquet,
    publish_text_annotations_metadata_parquet,
    save_working_quality_parquet,
    workflow_quality_parquet_path,
)

__all__ = [
    # features
    "ACTION_FIELD_CANDIDATES",
    "FRAME_INDEX_FIELD_CANDIDATES",
    "STATE_FIELD_CANDIDATES",
    "TASK_FIELD_CANDIDATES",
    "TIMESTAMP_FIELD_CANDIDATES",
    "build_episode_feature_vector",
    "build_episode_sequence",
    "build_joint_trajectory_payload",
    "clamp",
    "coerce_vector",
    "extract_joint_names",
    "first_present_value",
    "mean",
    "normalize_joint_names",
    "normalize_scalar_series",
    "percentile",
    "resolve_action_vector",
    "resolve_frame_index",
    "resolve_state_vector",
    "resolve_task_value",
    "resolve_timestamp",
    "sample_indices",
    "sample_sequence",
    "stdev",
    "summarize_series",
    # dtw
    "CARTESIAN_20D_GROUP_WEIGHTS",
    "CARTESIAN_20D_WINDOW_RATIO",
    "DEFAULT_DTW_HUBER_DELTA",
    "average_vectors",
    "build_distance_matrix",
    "build_distance_matrix_with_progress",
    "dtw_alignment",
    "dtw_distance",
    "euclidean_distance",
    "grouped_huber_distance",
    "huber_loss",
    "resolve_dtw_configuration",
    "vector_distance",
    # clustering
    "compute_dba_barycenter",
    "discover_prototype_clusters",
    "refine_clusters_with_dba",
    # canonical
    "ALOHA_ARM_JOINT_ORDER",
    "CANONICAL_GROUP_SLICES",
    "CanonicalTrajectory",
    "build_canonical_trajectory",
    "build_cartesian_canonical_trajectory",
    "build_cartesian_feature_rows",
    "build_joint_canonical_trajectory",
    # propagation
    "build_confidence_payload",
    "build_hf_annotation_rows",
    "build_phase_progress",
    "derive_quality_tags",
    "detect_grasp_place_events",
    "propagate_annotation_spans",
    # validators
    "VALIDATOR_REGISTRY",
    "finalize_validator",
    "load_episode_data",
    "make_issue",
    "run_quality_validators",
    "validate_action",
    "validate_depth_assets",
    "validate_metadata",
    "validate_timing",
    "validate_visual_assets",
    # state
    "init_workflow_state",
    "load_annotations",
    "load_propagation_results",
    "load_prototype_results",
    "load_quality_results",
    "load_workflow_state",
    "save_annotations",
    "save_propagation_results",
    "save_prototype_results",
    "save_quality_results",
    "save_workflow_state",
    # exports
    "build_quality_result_rows",
    "build_text_annotation_rows",
    "dataset_quality_parquet_path",
    "dataset_text_annotations_parquet_path",
    "export_quality_csv",
    "publish_quality_metadata_parquet",
    "publish_text_annotations_metadata_parquet",
    "save_working_quality_parquet",
    "workflow_quality_parquet_path",
]
