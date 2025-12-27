"""
Interpolation functions for rotations and transforms.

Provides spherical linear interpolation (SLERP), normalized linear interpolation (NLERP),
and transform sequence interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, overload

import numpy as np
import torch

from ._core import ArrayLike, EPS, SMALL_ANGLE_THRESHOLD, get_backend
from .rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

if TYPE_CHECKING:
    from .transform import Transform
    from ._core import UnitMismatchError


# =============================================================================
# Quaternion Interpolation
# =============================================================================


@overload
def quaternion_slerp(
    q0: np.ndarray, q1: np.ndarray, t: Union[float, np.ndarray]
) -> np.ndarray: ...
@overload
def quaternion_slerp(
    q0: torch.Tensor, q1: torch.Tensor, t: Union[float, torch.Tensor]
) -> torch.Tensor: ...


def quaternion_slerp(
    q0: ArrayLike,
    q1: ArrayLike,
    t: Union[float, ArrayLike],
) -> ArrayLike:
    """
    Spherical linear interpolation between quaternions.

    Interpolates along the shortest path on the unit quaternion sphere.

    Args:
        q0: Start quaternion(s) in xyzw format (..., 4)
        q1: End quaternion(s) in xyzw format (..., 4)
        t: Interpolation parameter(s) in [0, 1]. t=0 returns q0, t=1 returns q1.

    Returns:
        Interpolated quaternion(s) (..., 4)
    """
    backend = get_backend(q0)

    if backend == "numpy":
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t)

        q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)

        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        dot = np.abs(dot)
        dot = np.clip(dot, -1.0, 1.0)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        if np.ndim(t) == 0 or (np.ndim(t) == 1 and t.shape[0] == 1):
            t = np.broadcast_to(t, q0.shape[:-1] + (1,))
        elif t.ndim == 1 and q0.ndim == 1:
            t = np.expand_dims(t, axis=-1)
        elif t.ndim < q0.ndim:
            t = np.expand_dims(t, axis=-1)

        small_angle = np.abs(sin_theta) < SMALL_ANGLE_THRESHOLD
        safe_sin_theta = np.where(small_angle, 1.0, sin_theta)

        s0 = np.sin((1 - t) * theta) / safe_sin_theta
        s1 = np.sin(t * theta) / safe_sin_theta

        s0 = np.where(small_angle, 1 - t, s0)
        s1 = np.where(small_angle, t, s1)

        result = s0 * q0 + s1 * q1
        return result / np.linalg.norm(result, axis=-1, keepdims=True)

    # PyTorch implementation
    q0 = q0.clone()
    q1 = q1.clone()

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)

    if t.ndim == 1 and q0.ndim == 1:
        t = t.unsqueeze(-1)
    else:
        while t.ndim < q0.ndim:
            t = t.unsqueeze(-1)

    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    small_angle = sin_theta.abs() < SMALL_ANGLE_THRESHOLD

    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    s0 = torch.where(small_angle, 1 - t, s0)
    s1 = torch.where(small_angle, t, s1)

    result = s0 * q0 + s1 * q1
    return result / torch.norm(result, dim=-1, keepdim=True)


@overload
def quaternion_nlerp(
    q0: np.ndarray, q1: np.ndarray, t: Union[float, np.ndarray]
) -> np.ndarray: ...
@overload
def quaternion_nlerp(
    q0: torch.Tensor, q1: torch.Tensor, t: Union[float, torch.Tensor]
) -> torch.Tensor: ...


def quaternion_nlerp(
    q0: ArrayLike,
    q1: ArrayLike,
    t: Union[float, ArrayLike],
) -> ArrayLike:
    """
    Normalized linear interpolation between quaternions.

    Faster but less accurate than SLERP.

    Args:
        q0: Start quaternion(s) in xyzw format (..., 4)
        q1: End quaternion(s) in xyzw format (..., 4)
        t: Interpolation parameter(s) in [0, 1]

    Returns:
        Interpolated quaternion(s) (..., 4)
    """
    backend = get_backend(q0)

    if backend == "numpy":
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t)

        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
            t = np.broadcast_to(t, q0.shape[:-1] + (1,))
        elif t.ndim == 1 and q0.ndim == 1:
            t = np.expand_dims(t, axis=-1)
        elif t.ndim < q0.ndim:
            t = np.expand_dims(t, axis=-1)

        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)

        result = (1 - t) * q0 + t * q1
        return result / np.linalg.norm(result, axis=-1, keepdims=True)

    # PyTorch
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)

    if t.ndim == 1 and q0.ndim == 1:
        t = t.unsqueeze(-1)
    else:
        while t.ndim < q0.ndim:
            t = t.unsqueeze(-1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)

    result = (1 - t) * q0 + t * q1
    return result / torch.norm(result, dim=-1, keepdim=True)


# =============================================================================
# Transform Interpolation
# =============================================================================


def transform_interpolate(
    tf0: "Transform",
    tf1: "Transform",
    t: Union[float, ArrayLike],
    rotation_method: Literal["slerp", "nlerp"] = "slerp",
) -> "Transform":
    """
    Interpolate between two transforms.

    Uses linear interpolation for translation and extra,
    and spherical interpolation for rotation.

    Args:
        tf0: Start transform
        tf1: End transform (must have same translation_unit as tf0)
        t: Interpolation parameter(s) in [0, 1]
        rotation_method: "slerp" (accurate) or "nlerp" (fast)

    Returns:
        Interpolated Transform
    """
    from .transform import Transform
    from ._core import UnitMismatchError

    if tf0.backend != tf1.backend:
        raise ValueError(
            f"Cannot interpolate transforms with different backends: "
            f"{tf0.backend} vs {tf1.backend}"
        )

    if tf0.translation_unit != tf1.translation_unit:
        raise UnitMismatchError(
            f"Cannot interpolate transforms with different translation units: "
            f"{tf0.translation_unit.value} vs {tf1.translation_unit.value}."
        )

    backend = tf0.backend

    if backend == "torch":
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=tf0.rotation.dtype, device=tf0.rotation.device)
        t_trans = t
        if t_trans.ndim == 1 and tf0.translation.ndim == 1:
            t_trans = t_trans.unsqueeze(-1)
        else:
            while t_trans.ndim < tf0.translation.ndim:
                t_trans = t_trans.unsqueeze(-1)
    else:
        t = np.asarray(t)
        t_trans = t
        if t_trans.ndim == 1 and tf0.translation.ndim == 1:
            t_trans = np.expand_dims(t_trans, axis=-1)
        elif t_trans.ndim < tf0.translation.ndim:
            t_trans = np.expand_dims(t, axis=-1)

    translation = (1 - t_trans) * tf0.translation + t_trans * tf1.translation

    extra = None
    if tf0.extra is not None and tf1.extra is not None:
        extra = (1 - t_trans) * tf0.extra + t_trans * tf1.extra
    elif tf0.extra is not None:
        extra = tf0.extra
    elif tf1.extra is not None:
        extra = tf1.extra

    q0 = matrix_to_quaternion(tf0.rotation)
    q1 = matrix_to_quaternion(tf1.rotation)

    if rotation_method == "slerp":
        q_interp = quaternion_slerp(q0, q1, t)
    elif rotation_method == "nlerp":
        q_interp = quaternion_nlerp(q0, q1, t)
    else:
        raise ValueError(f"Unknown rotation_method: {rotation_method}")

    rotation = quaternion_to_matrix(q_interp)

    return Transform(
        rotation=rotation,
        translation=translation,
        translation_unit=tf0.translation_unit,
        extra=extra,
    )


def transform_sequence_interpolate(
    transforms: "Transform",
    times: ArrayLike,
    query_times: ArrayLike,
    rotation_method: Literal["slerp", "nlerp"] = "slerp",
    extrapolate: bool = False,
) -> "Transform":
    """
    Interpolate a sequence of transforms at query times (vectorized).

    Args:
        transforms: Batched Transform with shape (N, ...) containing keyframes
        times: Times corresponding to each transform (N,)
        query_times: Times at which to interpolate (M,)
        rotation_method: "slerp" or "nlerp"
        extrapolate: If True, allow extrapolation beyond time range

    Returns:
        Transform with batch dimension (M, ...)
    """
    from .transform import Transform

    n_keyframes = transforms.rotation.shape[0]
    if n_keyframes < 2:
        raise ValueError("Need at least 2 transforms for interpolation")

    backend = transforms.backend
    dtype = transforms.rotation.dtype
    device = transforms.device if backend == "torch" else None

    all_rot = transforms.rotation
    all_trans = transforms.translation
    all_extra = transforms.extra

    if backend == "numpy":
        times = np.asarray(times)
        query_times = np.asarray(query_times)

        if n_keyframes != len(times):
            raise ValueError(
                f"transforms ({n_keyframes}) and times ({len(times)}) must have same length"
            )

        all_quat = matrix_to_quaternion(all_rot)

        indices = np.searchsorted(times, query_times, side="right") - 1
        indices = np.clip(indices, 0, n_keyframes - 2)

        t0 = times[indices]
        t1 = times[indices + 1]

        alpha = (query_times - t0) / (t1 - t0 + EPS)
        if not extrapolate:
            alpha = np.clip(alpha, 0, 1)

        q0 = all_quat[indices]
        q1 = all_quat[indices + 1]
        trans0 = all_trans[indices]
        trans1 = all_trans[indices + 1]

        alpha_trans = alpha[:, np.newaxis]
        translation = (1 - alpha_trans) * trans0 + alpha_trans * trans1

        extra = None
        if all_extra is not None:
            extra0 = all_extra[indices]
            extra1 = all_extra[indices + 1]
            extra = (1 - alpha_trans) * extra0 + alpha_trans * extra1

        if rotation_method == "slerp":
            q_interp = quaternion_slerp(q0, q1, alpha)
        else:
            q_interp = quaternion_nlerp(q0, q1, alpha)

        rotation = quaternion_to_matrix(q_interp)

    else:
        # PyTorch
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times, dtype=dtype, device=device)
        if not isinstance(query_times, torch.Tensor):
            query_times = torch.tensor(query_times, dtype=dtype, device=device)

        if n_keyframes != len(times):
            raise ValueError(
                f"transforms ({n_keyframes}) and times ({len(times)}) must have same length"
            )

        all_quat = matrix_to_quaternion(all_rot)

        indices = torch.searchsorted(times, query_times, side="right") - 1
        indices = torch.clamp(indices, 0, n_keyframes - 2)

        t0 = times[indices]
        t1 = times[indices + 1]

        alpha = (query_times - t0) / (t1 - t0 + EPS)
        if not extrapolate:
            alpha = torch.clamp(alpha, 0, 1)

        q0 = all_quat[indices]
        q1 = all_quat[indices + 1]
        trans0 = all_trans[indices]
        trans1 = all_trans[indices + 1]

        alpha_trans = alpha.unsqueeze(-1)
        translation = (1 - alpha_trans) * trans0 + alpha_trans * trans1

        extra = None
        if all_extra is not None:
            extra0 = all_extra[indices]
            extra1 = all_extra[indices + 1]
            extra = (1 - alpha_trans) * extra0 + alpha_trans * extra1

        if rotation_method == "slerp":
            q_interp = quaternion_slerp(q0, q1, alpha)
        else:
            q_interp = quaternion_nlerp(q0, q1, alpha)

        rotation = quaternion_to_matrix(q_interp)

    return Transform(
        rotation=rotation,
        translation=translation,
        translation_unit=transforms.translation_unit,
        extra=extra,
    )

