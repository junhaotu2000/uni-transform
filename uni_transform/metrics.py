"""
Distance metrics for rotations and transforms.

Provides geodesic distance, translation distance, and transform distance functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple, Union, overload

import numpy as np
import torch

from ._core import ArrayLike, get_backend, matmul, transpose_last_two

if TYPE_CHECKING:
    from .transform import Transform


# =============================================================================
# Rotation Metrics
# =============================================================================


@overload
def geodesic_distance(
    R1: np.ndarray, R2: np.ndarray, reduce: Literal[True] = ..., degrees: bool = ...
) -> float: ...
@overload
def geodesic_distance(
    R1: np.ndarray, R2: np.ndarray, reduce: Literal[False] = ..., degrees: bool = ...
) -> np.ndarray: ...
@overload
def geodesic_distance(
    R1: torch.Tensor, R2: torch.Tensor, reduce: Literal[True] = ..., degrees: bool = ...
) -> float: ...
@overload
def geodesic_distance(
    R1: torch.Tensor, R2: torch.Tensor, reduce: Literal[False] = ..., degrees: bool = ...
) -> torch.Tensor: ...


def geodesic_distance(
    R1: ArrayLike,
    R2: ArrayLike,
    reduce: bool = True,
    degrees: bool = False,
) -> Union[float, ArrayLike]:
    """
    Compute geodesic distance (rotation angle) between rotation matrices.

    Args:
        R1: First rotation matrix (..., 3, 3)
        R2: Second rotation matrix (..., 3, 3)
        reduce: If True, return mean distance as float
        degrees: If True, return angle in degrees

    Returns:
        If reduce=True: float (mean angle)
        If reduce=False: array of angles with same shape as batch dims
    """
    backend = get_backend(R1)

    R_diff = matmul(transpose_last_two(R1), R2)

    if backend == "torch":
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        angle = torch.acos(cos_angle)

        if degrees:
            angle = torch.rad2deg(angle)

        return angle.mean().item() if reduce else angle

    trace = np.trace(R_diff, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if degrees:
        angle = np.rad2deg(angle)

    return float(np.mean(angle)) if reduce else angle


# =============================================================================
# Translation Metrics
# =============================================================================


@overload
def translation_distance(
    t1: np.ndarray, t2: np.ndarray, reduce: Literal[True] = ..., p: float = ...
) -> float: ...
@overload
def translation_distance(
    t1: np.ndarray, t2: np.ndarray, reduce: Literal[False] = ..., p: float = ...
) -> np.ndarray: ...
@overload
def translation_distance(
    t1: torch.Tensor, t2: torch.Tensor, reduce: Literal[True] = ..., p: float = ...
) -> float: ...
@overload
def translation_distance(
    t1: torch.Tensor, t2: torch.Tensor, reduce: Literal[False] = ..., p: float = ...
) -> torch.Tensor: ...


def translation_distance(
    t1: ArrayLike,
    t2: ArrayLike,
    reduce: bool = True,
    p: float = 2.0,
) -> Union[float, ArrayLike]:
    """
    Compute distance between translation vectors.

    Args:
        t1: First translation vector(s) (..., 3)
        t2: Second translation vector(s) (..., 3)
        reduce: If True, return mean distance as float
        p: Norm type (1.0 for L1, 2.0 for L2/Euclidean)

    Returns:
        If reduce=True: float (mean distance)
        If reduce=False: array of distances with same shape as batch dims
    """
    backend = get_backend(t1)

    if backend == "torch":
        diff = t1 - t2
        if p == 2.0:
            dist = torch.norm(diff, dim=-1)
        elif p == 1.0:
            dist = torch.abs(diff).sum(dim=-1)
        else:
            dist = torch.norm(diff, p=p, dim=-1)

        return dist.mean().item() if reduce else dist

    diff = t1 - t2
    if p == 2.0:
        dist = np.linalg.norm(diff, axis=-1)
    elif p == 1.0:
        dist = np.abs(diff).sum(axis=-1)
    else:
        dist = np.linalg.norm(diff, ord=p, axis=-1)

    return float(np.mean(dist)) if reduce else dist


# =============================================================================
# Transform Metrics
# =============================================================================


@overload
def transform_distance(
    tf1: "Transform",
    tf2: "Transform",
    reduce: Literal[True] = ...,
    rotation_weight: float = ...,
    translation_weight: float = ...,
    extra_weight: float = ...,
    degrees: bool = ...,
) -> Tuple[float, float, float, float]: ...
@overload
def transform_distance(
    tf1: "Transform",
    tf2: "Transform",
    reduce: Literal[False] = ...,
    rotation_weight: float = ...,
    translation_weight: float = ...,
    extra_weight: float = ...,
    degrees: bool = ...,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: ...


def transform_distance(
    tf1: "Transform",
    tf2: "Transform",
    reduce: bool = True,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
    extra_weight: float = 1.0,
    degrees: bool = False,
) -> Union[
    Tuple[float, float, float, float],
    Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
]:
    """
    Compute distance between transforms (rotation + translation + extra).

    Both transforms must have the same translation unit.

    Args:
        tf1: First transform
        tf2: Second transform (must have same translation_unit as tf1)
        reduce: If True, return mean distances as floats
        rotation_weight: Weight for rotation loss in total
        translation_weight: Weight for translation loss in total
        extra_weight: Weight for extra loss in total
        degrees: If True, return rotation angle in degrees

    Returns:
        Tuple of (total_loss, rotation_loss, translation_loss, extra_loss)
    """
    from ._core import UnitMismatchError

    if tf1.translation_unit != tf2.translation_unit:
        raise UnitMismatchError(
            f"Cannot compute distance between transforms with different translation units: "
            f"{tf1.translation_unit.value} vs {tf2.translation_unit.value}."
        )

    rot_dist = geodesic_distance(tf1.rotation, tf2.rotation, reduce=False, degrees=degrees)
    trans_dist = translation_distance(tf1.translation, tf2.translation, reduce=False)

    backend = tf1.backend

    if tf1.extra is not None and tf2.extra is not None:
        if backend == "torch":
            extra_dist = torch.norm(tf1.extra - tf2.extra, dim=-1)
        else:
            extra_dist = np.linalg.norm(tf1.extra - tf2.extra, axis=-1)
    else:
        if backend == "torch":
            extra_dist = torch.zeros_like(rot_dist)
        else:
            extra_dist = np.zeros_like(rot_dist)

    if backend == "torch":
        total = (
            rotation_weight * rot_dist
            + translation_weight * trans_dist
            + extra_weight * extra_dist
        )
        if reduce:
            return (
                total.mean().item(),
                rot_dist.mean().item(),
                trans_dist.mean().item(),
                extra_dist.mean().item(),
            )
        return total, rot_dist, trans_dist, extra_dist

    total = (
        rotation_weight * rot_dist
        + translation_weight * trans_dist
        + extra_weight * extra_dist
    )
    if reduce:
        return (
            float(np.mean(total)),
            float(np.mean(rot_dist)),
            float(np.mean(trans_dist)),
            float(np.mean(extra_dist)),
        )
    return total, rot_dist, trans_dist, extra_dist



