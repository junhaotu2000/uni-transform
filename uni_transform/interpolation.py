"""
Interpolation Module for uni_transform.

Unified API:
    interpolate(start, end, t, method="linear")     # Two-point
    interpolate_sequence(points, times, query, ...) # Multi-point

Supported methods by type:
    - Vector/Scalar: "linear", "minimum_jerk", "cubic_spline"
    - Quaternion: "slerp", "nlerp", "squad" (multi-point only)
    - Rotation: "slerp", "nlerp"
    - Transform: rotation_method + translation_method combination

Low-level functions also available:
    - quaternion_slerp, quaternion_nlerp, quaternion_squad
    - minimum_jerk_interpolate, minimum_jerk_velocity, minimum_jerk_acceleration
    - compute_spline, cubic_spline_interpolate, cubic_spline_derivative
    - transform_interpolate, transform_sequence_interpolate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union, overload

import numpy as np
import torch

from ._core import (
    ArrayLike,
    Backend,
    EPS,
    QuatInterpMethod,
    RotationSeqInterpMethod,
    SMALL_ANGLE_THRESHOLD,
    UnitMismatchError,
    VectorInterpMethod,
    get_backend,
)
from .rotation import Rotation
from .transform import Transform
from .rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_conjugate,
    quaternion_multiply,
)

# =============================================================================
# JIT-compiled PyTorch kernels (hot paths)
# =============================================================================

# Flag to enable/disable JIT (useful for debugging)
_USE_JIT = True

try:
    @torch.jit.script
    def _slerp_kernel(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, 
                      small_thresh: float, eps: float) -> torch.Tensor:
        """JIT-compiled SLERP kernel."""
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
        
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        small = sin_theta.abs() < small_thresh
        s0 = torch.where(small, 1 - t, torch.sin((1 - t) * theta) / (sin_theta + eps))
        s1 = torch.where(small, t, torch.sin(t * theta) / (sin_theta + eps))
        
        result = s0 * q0 + s1 * q1
        return result / torch.norm(result, dim=-1, keepdim=True)

    @torch.jit.script
    def _nlerp_kernel(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """JIT-compiled NLERP kernel."""
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        result = (1 - t) * q0 + t * q1
        return result / torch.norm(result, dim=-1, keepdim=True)

    @torch.jit.script
    def _min_jerk_kernel(c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor,
                         c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """JIT-compiled minimum jerk polynomial (Horner's method)."""
        return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))

    _JIT_AVAILABLE = True
except (RuntimeError, AttributeError):  # JIT compilation or missing torch.jit
    _JIT_AVAILABLE = False
    

# =============================================================================
# SplineCoefficients
# =============================================================================


@dataclass(slots=True)
class SplineCoefficients:
    """Cubic spline coefficients for reuse.

    Compute once with compute_spline(), evaluate many times with evaluate().

    For segment i: S_i(t) = a[i] + b[i]*(t-t_i) + c[i]*(t-t_i)² + d[i]*(t-t_i)³

    Example:
        >>> spline = compute_spline(points, times)
        >>> pos = spline.evaluate(query_times)
        >>> vel = spline.derivative(query_times, order=1)
    """

    a: ArrayLike  # (N-1, D)
    b: ArrayLike  # (N-1, D)
    c: ArrayLike  # (N-1, D)
    d: ArrayLike  # (N-1, D)
    times: ArrayLike  # (N,)
    backend: Backend = field(init=False)

    def __post_init__(self) -> None:
        self.backend = get_backend(self.a)

    @property
    def n_segments(self) -> int:
        return self.a.shape[0] if self.a.ndim > 0 else 1

    @property
    def n_dims(self) -> int:
        return self.a.shape[-1] if self.a.ndim > 1 else 1

    def evaluate(self, query_times: ArrayLike) -> ArrayLike:
        """Evaluate spline at query times."""
        return _evaluate_spline(self, query_times)

    def derivative(self, query_times: ArrayLike, order: int = 1) -> ArrayLike:
        """Compute derivative (order=1,2,3) at query times."""
        return _evaluate_spline_derivative(self, query_times, order)


# =============================================================================
# Unified API
# =============================================================================


def interpolate(
    start: ArrayLike,
    end: ArrayLike,
    t: Union[float, ArrayLike],
    method: VectorInterpMethod = "linear",
    duration: float = 1.0,
) -> ArrayLike:
    """
    Unified two-point interpolation for vectors/scalars.

    Args:
        start: Start value (..., D)
        end: End value (..., D)
        t: Time parameter(s). For "linear": t in [0,1]. For others: t in [0, duration]
        method: "linear", "minimum_jerk", or "cubic_spline"
        duration: Total duration (for minimum_jerk)

    Returns:
        Interpolated value(s)

    Example:
        >>> pos = interpolate(start, end, t=0.5)  # Linear
        >>> pos = interpolate(start, end, t=0.5, method="minimum_jerk", duration=2.0)
    """
    if method == "linear":
        return _linear_interpolate(start, end, t)
    elif method == "minimum_jerk":
        return minimum_jerk_interpolate(start, end, t, duration)
    elif method == "cubic_spline":
        # For two points, cubic spline reduces to linear
        return _linear_interpolate(start, end, t)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'minimum_jerk', or 'cubic_spline'")


def interpolate_sequence(
    points: ArrayLike,
    times: ArrayLike,
    query_times: ArrayLike,
    method: VectorInterpMethod = "linear",
) -> ArrayLike:
    """
    Unified multi-point interpolation for vectors/scalars.

    Args:
        points: Keyframe values (N, D)
        times: Time for each keyframe (N,)
        query_times: Times at which to interpolate (M,)
        method: "linear", "minimum_jerk", or "cubic_spline"

    Returns:
        Interpolated values (M, D)

    Example:
        >>> positions = interpolate_sequence(waypoints, times, query_times)
        >>> positions = interpolate_sequence(waypoints, times, query_times, method="cubic_spline")
    """
    if method == "linear":
        return _linear_sequence_interpolate(points, times, query_times)
    elif method == "minimum_jerk":
        return _minimum_jerk_sequence_interpolate(points, times, query_times)
    elif method == "cubic_spline":
        return cubic_spline_interpolate(points, times, query_times)
    else:
        raise ValueError(f"Unknown method: {method}")


def interpolate_rotation(
    r0: "Rotation",
    r1: "Rotation",
    t: Union[float, ArrayLike],
    method: QuatInterpMethod = "slerp",
) -> "Rotation":
    """
    Two-point rotation interpolation.

    Args:
        r0, r1: Start and end Rotation objects
        t: Parameter in [0, 1]
        method: "slerp" (accurate) or "nlerp" (fast)

    Returns:
        Interpolated Rotation
    """
    q0 = matrix_to_quaternion(r0.matrix)
    q1 = matrix_to_quaternion(r1.matrix)

    if method == "slerp":
        q = quaternion_slerp(q0, q1, t)
    else:
        q = quaternion_nlerp(q0, q1, t)

    return Rotation(matrix=quaternion_to_matrix(q))


def interpolate_rotation_sequence(
    rotations: "Rotation",
    times: ArrayLike,
    query_times: ArrayLike,
    method: Union[RotationSeqInterpMethod, str] = "slerp",
) -> "Rotation":
    """
    Multi-point rotation interpolation.

    Args:
        rotations: Batched Rotation (N, 3, 3)
        times: Time for each keyframe (N,)
        query_times: Query times (M,)
        method: "slerp", "nlerp" (piecewise), or "squad" (smooth)

    Returns:
        Interpolated Rotation (M, 3, 3)
    """
    all_quat = matrix_to_quaternion(rotations.matrix)

    if method == "squad":
        q = quaternion_squad(all_quat, times, query_times)
    else:
        q = _piecewise_quat_interpolate(all_quat, times, query_times, method)

    return Rotation(matrix=quaternion_to_matrix(q))


def interpolate_transform(
    tf0: "Transform",
    tf1: "Transform",
    t: Union[float, ArrayLike],
    rotation_method: QuatInterpMethod = "slerp",
    translation_method: VectorInterpMethod = "linear",
    extra_method: VectorInterpMethod = "linear",
    duration: float = 1.0,
) -> "Transform":
    """
    Two-point transform interpolation with configurable methods.

    Args:
        tf0, tf1: Start and end transforms
        t: Time parameter
        rotation_method: "slerp" or "nlerp"
        translation_method: "linear" or "minimum_jerk"
        extra_method: "linear" or "minimum_jerk" for extra dimensions (e.g., gripper_width)
        duration: Duration for minimum_jerk

    Returns:
        Interpolated Transform

    Example:
        >>> tf = interpolate_transform(tf0, tf1, t=0.5)
        >>> tf = interpolate_transform(tf0, tf1, t=0.5, 
        ...     rotation_method="nlerp", translation_method="minimum_jerk", duration=2.0)
        >>> # Smooth gripper interpolation
        >>> tf = interpolate_transform(tf0, tf1, t=0.5, extra_method="minimum_jerk", duration=1.0)
    """
    if tf0.backend != tf1.backend:
        raise ValueError(f"Backend mismatch: {tf0.backend} vs {tf1.backend}")
    if tf0.translation_unit != tf1.translation_unit:
        raise UnitMismatchError(f"Unit mismatch: {tf0.translation_unit} vs {tf1.translation_unit}")

    # Translation
    if translation_method == "minimum_jerk":
        translation = minimum_jerk_interpolate(tf0.translation, tf1.translation, t, duration)
        # Normalize t for rotation
        backend = tf0.backend
        if backend == "torch":
            t_rot = t / duration if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=tf0.rotation.dtype) / duration
        else:
            t_rot = np.asarray(t) / duration
    else:
        translation = _linear_interpolate(tf0.translation, tf1.translation, t)
        t_rot = t

    # Rotation
    q0 = matrix_to_quaternion(tf0.rotation)
    q1 = matrix_to_quaternion(tf1.rotation)
    q = quaternion_slerp(q0, q1, t_rot) if rotation_method == "slerp" else quaternion_nlerp(q0, q1, t_rot)
    rotation = quaternion_to_matrix(q)

    # Extra (with configurable interpolation method)
    extra = _interpolate_extra(tf0.extra, tf1.extra, t, extra_method, duration)

    return Transform(rotation=rotation, translation=translation, translation_unit=tf0.translation_unit, extra=extra)


def interpolate_transform_sequence(
    transforms: "Transform",
    times: ArrayLike,
    query_times: ArrayLike,
    rotation_method: Union[RotationSeqInterpMethod, str] = "slerp",
    translation_method: Union[VectorInterpMethod, str] = "linear",
    extra_method: Union[VectorInterpMethod, str] = "linear",
    extrapolate: bool = False,
) -> "Transform":
    """
    Multi-point transform interpolation with configurable methods.

    Args:
        transforms: Batched Transform (N, ...)
        times: Keyframe times (N,)
        query_times: Query times (M,)
        rotation_method: "slerp", "nlerp", or "squad"
        translation_method: "linear", "minimum_jerk", or "cubic_spline"
        extra_method: "linear", "minimum_jerk", or "cubic_spline" for extra (e.g., gripper_width)
        extrapolate: Allow extrapolation

    Returns:
        Interpolated Transform (M, ...)

    Example:
        >>> result = interpolate_transform_sequence(
        ...     keyframes, times, query_times,
        ...     rotation_method="squad",
        ...     translation_method="cubic_spline",
        ...     extra_method="cubic_spline"  # Smooth gripper trajectory
        ... )
    """
    backend = transforms.backend
    n = transforms.rotation.shape[0]

    if n < 2:
        raise ValueError("Need at least 2 transforms")

    # Translation
    translation = interpolate_sequence(transforms.translation, times, query_times, translation_method)

    # Rotation
    all_quat = matrix_to_quaternion(transforms.rotation)
    if rotation_method == "squad":
        q = quaternion_squad(all_quat, times, query_times)
    else:
        q = _piecewise_quat_interpolate(all_quat, times, query_times, rotation_method, extrapolate)
    rotation = quaternion_to_matrix(q)

    # Extra (with configurable interpolation method)
    extra = None
    if transforms.extra is not None:
        extra = interpolate_sequence(transforms.extra, times, query_times, extra_method)

    return Transform(rotation=rotation, translation=translation, translation_unit=transforms.translation_unit, extra=extra)


# =============================================================================
# Quaternion Interpolation
# =============================================================================


@overload
def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray: ...
@overload
def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor: ...


def quaternion_slerp(q0: ArrayLike, q1: ArrayLike, t: Union[float, ArrayLike]) -> ArrayLike:
    """
    Spherical linear interpolation between quaternions.

    Args:
        q0, q1: Quaternions in xyzw format (..., 4)
        t: Parameter in [0, 1]

    Returns:
        Interpolated quaternion (..., 4)
    """
    backend = get_backend(q0)

    if backend == "numpy":
        return _slerp_numpy(np.asarray(q0), np.asarray(q1), np.asarray(t))
    return _slerp_torch(q0, q1, t)


@overload
def quaternion_nlerp(q0: np.ndarray, q1: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray: ...
@overload
def quaternion_nlerp(q0: torch.Tensor, q1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor: ...


def quaternion_nlerp(q0: ArrayLike, q1: ArrayLike, t: Union[float, ArrayLike]) -> ArrayLike:
    """
    Normalized linear interpolation (faster than SLERP, good for small angles).

    Args:
        q0, q1: Quaternions in xyzw format (..., 4)
        t: Parameter in [0, 1]

    Returns:
        Interpolated quaternion (..., 4)
    """
    backend = get_backend(q0)

    if backend == "numpy":
        return _nlerp_numpy(np.asarray(q0), np.asarray(q1), np.asarray(t))
    return _nlerp_torch(q0, q1, t)


def quaternion_squad(
    quaternions: ArrayLike,
    times: ArrayLike,
    query_times: ArrayLike,
) -> ArrayLike:
    """
    SQUAD: Smooth multi-point quaternion interpolation (C1-continuous).

    Args:
        quaternions: Keyframe quaternions (N, 4) in xyzw format
        times: Keyframe times (N,)
        query_times: Query times (M,)

    Returns:
        Interpolated quaternions (M, 4)
    """
    backend = get_backend(quaternions)
    n = quaternions.shape[0]

    if n < 2:
        raise ValueError("Need at least 2 quaternions")

    if backend == "numpy":
        return _squad_numpy(quaternions, np.asarray(times), np.asarray(query_times))
    return _squad_torch(quaternions, times, query_times)


# =============================================================================
# Minimum Jerk Interpolation
# =============================================================================


def minimum_jerk_interpolate(
    start: ArrayLike,
    end: ArrayLike,
    t: Union[float, ArrayLike],
    duration: float = 1.0,
    start_velocity: Optional[ArrayLike] = None,
    end_velocity: Optional[ArrayLike] = None,
    start_acceleration: Optional[ArrayLike] = None,
    end_acceleration: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Minimum jerk (5th-order polynomial) interpolation.

    Creates smooth trajectory minimizing jerk. Widely used in robotics.

    Args:
        start, end: Start and end positions (..., D)
        t: Time(s) in [0, duration]
        duration: Total trajectory duration
        start_velocity, end_velocity: Boundary velocities (default: zeros)
        start_acceleration, end_acceleration: Boundary accelerations (default: zeros)

    Returns:
        Position(s) at time t
    """
    backend = get_backend(start)
    if backend == "numpy":
        return _min_jerk_numpy(start, end, t, duration, start_velocity, end_velocity, start_acceleration, end_acceleration)
    return _min_jerk_torch(start, end, t, duration, start_velocity, end_velocity, start_acceleration, end_acceleration)


def minimum_jerk_velocity(
    start: ArrayLike,
    end: ArrayLike,
    t: Union[float, ArrayLike],
    duration: float = 1.0,
    start_velocity: Optional[ArrayLike] = None,
    end_velocity: Optional[ArrayLike] = None,
    start_acceleration: Optional[ArrayLike] = None,
    end_acceleration: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Compute velocity along minimum jerk trajectory."""
    backend = get_backend(start)
    if backend == "numpy":
        return _min_jerk_deriv_numpy(start, end, t, duration, start_velocity, end_velocity, start_acceleration, end_acceleration, order=1)
    return _min_jerk_deriv_torch(start, end, t, duration, start_velocity, end_velocity, start_acceleration, end_acceleration, order=1)


def minimum_jerk_acceleration(
    start: ArrayLike,
    end: ArrayLike,
    t: Union[float, ArrayLike],
    duration: float = 1.0,
    start_velocity: Optional[ArrayLike] = None,
    end_velocity: Optional[ArrayLike] = None,
    start_acceleration: Optional[ArrayLike] = None,
    end_acceleration: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Compute acceleration along minimum jerk trajectory."""
    backend = get_backend(start)
    if backend == "numpy":
        return _min_jerk_deriv_numpy(start, end, t, duration, start_velocity, end_velocity, start_acceleration, end_acceleration, order=2)
    return _min_jerk_deriv_torch(start, end, t, duration, start_velocity, end_velocity, start_acceleration, end_acceleration, order=2)


# =============================================================================
# Cubic Spline Interpolation
# =============================================================================


def compute_spline(
    points: ArrayLike,
    times: ArrayLike,
    boundary: Literal["natural", "clamped", "not-a-knot"] = "natural",
    start_derivative: Optional[ArrayLike] = None,
    end_derivative: Optional[ArrayLike] = None,
) -> SplineCoefficients:
    """
    Compute cubic spline coefficients (compute once, evaluate many times).

    Args:
        points: Control points (N, D)
        times: Time values (N,)
        boundary: "natural" (zero curvature), "clamped" (specified derivatives), "not-a-knot"
        start_derivative, end_derivative: For "clamped" boundary

    Returns:
        SplineCoefficients with evaluate() and derivative() methods

    Example:
        >>> spline = compute_spline(waypoints, times)
        >>> pos = spline.evaluate(query_times)
        >>> vel = spline.derivative(query_times, order=1)
    """
    a, b, c, d = cubic_spline_coefficients(points, times, boundary, start_derivative, end_derivative)

    backend = get_backend(points)
    if backend == "numpy":
        times_arr = np.asarray(times)
    else:
        times_arr = times if isinstance(times, torch.Tensor) else torch.tensor(times, dtype=points.dtype, device=points.device)

    return SplineCoefficients(a=a, b=b, c=c, d=d, times=times_arr)


def cubic_spline_interpolate(
    points: ArrayLike,
    times: ArrayLike,
    query_times: ArrayLike,
    boundary: Literal["natural", "clamped", "not-a-knot"] = "natural",
    start_derivative: Optional[ArrayLike] = None,
    end_derivative: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Cubic spline interpolation (C2-continuous).

    Args:
        points: Control points (N, D)
        times: Time values (N,)
        query_times: Times to interpolate (M,)
        boundary: Boundary condition type
        start_derivative, end_derivative: For "clamped"

    Returns:
        Interpolated points (M, D)
    """
    spline = compute_spline(points, times, boundary, start_derivative, end_derivative)
    return spline.evaluate(query_times)


def cubic_spline_derivative(
    points: ArrayLike,
    times: ArrayLike,
    query_times: ArrayLike,
    boundary: Literal["natural", "clamped", "not-a-knot"] = "natural",
    start_derivative: Optional[ArrayLike] = None,
    end_derivative: Optional[ArrayLike] = None,
    order: int = 1,
) -> ArrayLike:
    """Compute derivative of cubic spline (order=1,2,3)."""
    spline = compute_spline(points, times, boundary, start_derivative, end_derivative)
    return spline.derivative(query_times, order)


def cubic_spline_coefficients(
    points: ArrayLike,
    times: ArrayLike,
    boundary: Literal["natural", "clamped", "not-a-knot"] = "natural",
    start_derivative: Optional[ArrayLike] = None,
    end_derivative: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute raw cubic spline coefficients.

    Returns (a, b, c, d) where S_i(t) = a[i] + b[i]*(t-t_i) + c[i]*(t-t_i)² + d[i]*(t-t_i)³
    """
    backend = get_backend(points)
    if backend == "numpy":
        return _spline_coeffs_numpy(points, times, boundary, start_derivative, end_derivative)
    return _spline_coeffs_torch(points, times, boundary, start_derivative, end_derivative)


# =============================================================================
# Internal: Linear Interpolation
# =============================================================================


def _linear_interpolate(start: ArrayLike, end: ArrayLike, t: Union[float, ArrayLike]) -> ArrayLike:
    """Linear interpolation: (1-t)*start + t*end
    
    Handles batched t: if t is (M,) and start is (D,), output is (M, D).
    """
    backend = get_backend(start)
    if backend == "numpy":
        t = np.asarray(t)
        # Handle case: t=(M,), start=(D,) -> output (M, D)
        if t.ndim == 1 and start.ndim == 1 and t.shape[0] != start.shape[0]:
            t = t[:, np.newaxis]  # (M, 1)
        elif t.ndim > 0 and t.ndim < start.ndim:
            t = t.reshape((-1,) + (1,) * (start.ndim - t.ndim + 1))
        return (1 - t) * start + t * end
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=start.dtype, device=start.device)
    # Handle case: t=(M,), start=(D,) -> output (M, D)
    if t.ndim == 1 and start.ndim == 1 and t.shape[0] != start.shape[0]:
        t = t.unsqueeze(-1)  # (M, 1)
    else:
        while t.ndim < start.ndim:
            t = t.unsqueeze(-1)
    return (1 - t) * start + t * end


def _linear_sequence_interpolate(points: ArrayLike, times: ArrayLike, query_times: ArrayLike) -> ArrayLike:
    """Piecewise linear interpolation for sequences."""
    backend = get_backend(points)
    n = points.shape[0]

    if backend == "numpy":
        times = np.asarray(times)
        query_times = np.asarray(query_times)
        indices = np.clip(np.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
        t0, t1 = times[indices], times[indices + 1]
        alpha = ((query_times - t0) / (t1 - t0 + EPS))[:, np.newaxis]
        return (1 - alpha) * points[indices] + alpha * points[indices + 1]

    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=points.dtype, device=points.device)
    if not isinstance(query_times, torch.Tensor):
        query_times = torch.tensor(query_times, dtype=points.dtype, device=points.device)
    indices = torch.clamp(torch.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
    t0, t1 = times[indices], times[indices + 1]
    alpha = ((query_times - t0) / (t1 - t0 + EPS)).unsqueeze(-1)
    return (1 - alpha) * points[indices] + alpha * points[indices + 1]


def _minimum_jerk_sequence_interpolate(points: ArrayLike, times: ArrayLike, query_times: ArrayLike) -> ArrayLike:
    """Vectorized piecewise minimum jerk interpolation for sequences (no loop)."""
    backend = get_backend(points)
    n = points.shape[0]

    if backend == "numpy":
        times = np.asarray(times)
        query_times = np.asarray(query_times)
        indices = np.clip(np.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
        
        # Vectorized: get all segment starts/ends at once
        p0, p1 = points[indices], points[indices + 1]  # (M, D)
        t0, t1 = times[indices], times[indices + 1]    # (M,)
        durations = t1 - t0 + EPS                       # (M,)
        local_t = query_times - t0                      # (M,)
        
        # Vectorized minimum jerk: compute all coefficients at once
        # Zero velocity/acceleration at segment boundaries (piecewise)
        v0 = v1 = a0 = a1 = np.zeros_like(p0)
        T = durations[:, np.newaxis]  # (M, 1)
        T2, T3, T4, T5 = T*T, T*T*T, T*T*T*T, T*T*T*T*T
        
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        c3 = (20*(p1 - p0) - (8*v1 + 12*v0)*T - (3*a0 - a1)*T2) / (2*T3 + EPS)
        c4 = (30*(p0 - p1) + (14*v1 + 16*v0)*T + (3*a0 - 2*a1)*T2) / (2*T4 + EPS)
        c5 = (12*(p1 - p0) - 6*(v1 + v0)*T - (a0 - a1)*T2) / (2*T5 + EPS)
        
        t_exp = local_t[:, np.newaxis]  # (M, 1)
        return c0 + c1*t_exp + c2*t_exp**2 + c3*t_exp**3 + c4*t_exp**4 + c5*t_exp**5

    # PyTorch backend
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=points.dtype, device=points.device)
    if not isinstance(query_times, torch.Tensor):
        query_times = torch.tensor(query_times, dtype=points.dtype, device=points.device)
    
    indices = torch.clamp(torch.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
    
    # Vectorized: get all segment starts/ends at once
    p0, p1 = points[indices], points[indices + 1]  # (M, D)
    t0, t1 = times[indices], times[indices + 1]    # (M,)
    durations = t1 - t0 + EPS                       # (M,)
    local_t = query_times - t0                      # (M,)
    
    # Vectorized minimum jerk
    v0 = v1 = a0 = a1 = torch.zeros_like(p0)
    T = durations.unsqueeze(-1)  # (M, 1)
    T2, T3, T4, T5 = T*T, T*T*T, T*T*T*T, T*T*T*T*T
    
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    c3 = (20*(p1 - p0) - (8*v1 + 12*v0)*T - (3*a0 - a1)*T2) / (2*T3 + EPS)
    c4 = (30*(p0 - p1) + (14*v1 + 16*v0)*T + (3*a0 - 2*a1)*T2) / (2*T4 + EPS)
    c5 = (12*(p1 - p0) - 6*(v1 + v0)*T - (a0 - a1)*T2) / (2*T5 + EPS)
    
    t_exp = local_t.unsqueeze(-1)  # (M, 1)
    
    # Use JIT kernel if available
    if _JIT_AVAILABLE and _USE_JIT:
        return _min_jerk_kernel(c0, c1, c2, c3, c4, c5, t_exp)
    
    return c0 + c1*t_exp + c2*t_exp**2 + c3*t_exp**3 + c4*t_exp**4 + c5*t_exp**5


def _interpolate_extra(
    extra0: Optional[ArrayLike],
    extra1: Optional[ArrayLike],
    t: Union[float, ArrayLike],
    method: VectorInterpMethod = "linear",
    duration: float = 1.0,
) -> Optional[ArrayLike]:
    """Interpolate extra data (e.g., gripper_width) with configurable method."""
    if extra0 is not None and extra1 is not None:
        if method == "linear":
            return _linear_interpolate(extra0, extra1, t)
        elif method == "minimum_jerk":
            return minimum_jerk_interpolate(extra0, extra1, t, duration)
        else:  # cubic_spline -> linear for two points
            return _linear_interpolate(extra0, extra1, t)
    return extra0 if extra0 is not None else extra1


def _piecewise_quat_interpolate(
    quaternions: ArrayLike,
    times: ArrayLike,
    query_times: ArrayLike,
    method: str = "slerp",
    extrapolate: bool = False,
) -> ArrayLike:
    """Piecewise quaternion interpolation (slerp or nlerp between segments)."""
    backend = get_backend(quaternions)
    n = quaternions.shape[0]

    if backend == "numpy":
        times = np.asarray(times)
        query_times = np.asarray(query_times)
        indices = np.clip(np.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
        t0, t1 = times[indices], times[indices + 1]
        alpha = (query_times - t0) / (t1 - t0 + EPS)
        if not extrapolate:
            alpha = np.clip(alpha, 0, 1)
        q0, q1 = quaternions[indices], quaternions[indices + 1]
        return quaternion_slerp(q0, q1, alpha) if method == "slerp" else quaternion_nlerp(q0, q1, alpha)

    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=quaternions.dtype, device=quaternions.device)
    if not isinstance(query_times, torch.Tensor):
        query_times = torch.tensor(query_times, dtype=quaternions.dtype, device=quaternions.device)
    indices = torch.clamp(torch.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
    t0, t1 = times[indices], times[indices + 1]
    alpha = (query_times - t0) / (t1 - t0 + EPS)
    if not extrapolate:
        alpha = torch.clamp(alpha, 0, 1)
    q0, q1 = quaternions[indices], quaternions[indices + 1]
    return quaternion_slerp(q0, q1, alpha) if method == "slerp" else quaternion_nlerp(q0, q1, alpha)


# =============================================================================
# Internal: SLERP/NLERP Implementation
# =============================================================================


def _slerp_numpy(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Handle batched t with single quaternion: q0=(4,), t=(M,) -> output (M, 4)
    single_quat = q0.ndim == 1
    if single_quat and t.ndim == 1 and t.shape[0] > 1:
        # Expand q0, q1 for batched interpolation
        q0 = q0[np.newaxis, :]  # (1, 4)
        q1 = q1[np.newaxis, :]  # (1, 4)
        t = t[:, np.newaxis]    # (M, 1)

    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    # Expand t for broadcasting if not already done
    if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
        t = np.broadcast_to(t, q0.shape[:-1] + (1,))
    elif t.ndim < q0.ndim:
        t = np.expand_dims(t, axis=-1)

    small = np.abs(sin_theta) < SMALL_ANGLE_THRESHOLD
    safe_sin = np.where(small, 1.0, sin_theta)

    s0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / safe_sin)
    s1 = np.where(small, t, np.sin(t * theta) / safe_sin)

    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _slerp_torch(q0: torch.Tensor, q1: torch.Tensor, t) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)

    # Handle batched t with single quaternion: q0=(4,), t=(M,) -> output (M, 4)
    single_quat = q0.ndim == 1
    if single_quat and t.ndim == 1 and t.shape[0] > 1:
        q0 = q0.unsqueeze(0)  # (1, 4)
        q1 = q1.unsqueeze(0)  # (1, 4)
        t = t.unsqueeze(-1)   # (M, 1)

    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)

    while t.ndim < q0.ndim:
        t = t.unsqueeze(-1)

    # Use JIT kernel if available
    if _JIT_AVAILABLE and _USE_JIT:
        return _slerp_kernel(q0, q1, t, SMALL_ANGLE_THRESHOLD, EPS)

    # Fallback implementation
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp(torch.abs(dot), -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    small = sin_theta.abs() < SMALL_ANGLE_THRESHOLD
    s0 = torch.where(small, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
    s1 = torch.where(small, t, torch.sin(t * theta) / sin_theta)

    result = s0 * q0 + s1 * q1
    return result / torch.norm(result, dim=-1, keepdim=True)


def _nlerp_numpy(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Handle batched t with single quaternion: q0=(4,), t=(M,) -> output (M, 4)
    single_quat = q0.ndim == 1
    if single_quat and t.ndim == 1 and t.shape[0] > 1:
        q0 = q0[np.newaxis, :]  # (1, 4)
        q1 = q1[np.newaxis, :]  # (1, 4)
        t = t[:, np.newaxis]    # (M, 1)
    elif t.ndim < q0.ndim:
        t = np.expand_dims(t, axis=-1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    result = (1 - t) * q0 + t * q1
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _nlerp_torch(q0: torch.Tensor, q1: torch.Tensor, t) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)
    # Handle batched t with single quaternion: q0=(4,), t=(M,) -> output (M, 4)
    single_quat = q0.ndim == 1
    if single_quat and t.ndim == 1 and t.shape[0] > 1:
        q0 = q0.unsqueeze(0)  # (1, 4)
        q1 = q1.unsqueeze(0)  # (1, 4)
        t = t.unsqueeze(-1)   # (M, 1)
    else:
        while t.ndim < q0.ndim:
            t = t.unsqueeze(-1)
    
    # Use JIT kernel if available
    if _JIT_AVAILABLE and _USE_JIT:
        return _nlerp_kernel(q0, q1, t)
    
    # Fallback implementation
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    result = (1 - t) * q0 + t * q1
    return result / torch.norm(result, dim=-1, keepdim=True)


# =============================================================================
# Internal: SQUAD Implementation
# =============================================================================


def _quat_exp(v: ArrayLike) -> ArrayLike:
    """Quaternion exponential of pure quaternion."""
    backend = get_backend(v)
    if backend == "numpy":
        theta = np.linalg.norm(v, axis=-1, keepdims=True)
        safe = np.where(theta < EPS, EPS, theta)
        xyz = v / safe * np.sin(theta)
        w = np.cos(theta)
        result = np.concatenate([xyz, w], axis=-1)
        identity = np.zeros_like(result)
        identity[..., 3] = 1.0
        return np.where(theta < EPS, identity, result)
    theta = torch.norm(v, dim=-1, keepdim=True)
    safe = torch.where(theta < EPS, torch.ones_like(theta) * EPS, theta)
    xyz = v / safe * torch.sin(theta)
    w = torch.cos(theta)
    result = torch.cat([xyz, w], dim=-1)
    identity = torch.zeros_like(result)
    identity[..., 3] = 1.0
    return torch.where(theta < EPS, identity, result)


def _quat_log(q: ArrayLike) -> ArrayLike:
    """Quaternion logarithm -> pure quaternion."""
    backend = get_backend(q)
    if backend == "numpy":
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        xyz, w = q[..., :3], np.clip(q[..., 3:4], -1.0, 1.0)
        norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
        safe = np.where(norm < EPS, EPS, norm)
        theta = np.arccos(w)
        result = theta * xyz / safe
        return np.where(norm < EPS, np.zeros_like(xyz), result)
    q = q / torch.norm(q, dim=-1, keepdim=True)
    xyz, w = q[..., :3], torch.clamp(q[..., 3:4], -1.0, 1.0)
    norm = torch.norm(xyz, dim=-1, keepdim=True)
    safe = torch.where(norm < EPS, torch.ones_like(norm) * EPS, norm)
    theta = torch.acos(w)
    result = theta * xyz / safe
    return torch.where(norm < EPS, torch.zeros_like(xyz), result)


# Use optimized versions from rotation_conversions
_quat_mul = quaternion_multiply
_quat_conj = quaternion_conjugate


def _squad_tangent(q_prev: ArrayLike, q_curr: ArrayLike, q_next: ArrayLike) -> ArrayLike:
    """Compute SQUAD tangent quaternion."""
    backend = get_backend(q_curr)
    q_inv = _quat_conj(q_curr)

    # Ensure shortest path
    if backend == "numpy":
        dot_next = (q_curr * q_next).sum(axis=-1, keepdims=True)
        dot_prev = (q_curr * q_prev).sum(axis=-1, keepdims=True)
        q_next = np.where(dot_next < 0, -q_next, q_next)
        q_prev = np.where(dot_prev < 0, -q_prev, q_prev)
    else:
        dot_next = (q_curr * q_next).sum(dim=-1, keepdim=True)
        dot_prev = (q_curr * q_prev).sum(dim=-1, keepdim=True)
        q_next = torch.where(dot_next < 0, -q_next, q_next)
        q_prev = torch.where(dot_prev < 0, -q_prev, q_prev)

    log_next = _quat_log(_quat_mul(q_inv, q_next))
    log_prev = _quat_log(_quat_mul(q_inv, q_prev))
    tangent = _quat_exp(-0.25 * (log_next + log_prev))
    return _quat_mul(q_curr, tangent)


def _squad_tangents_vectorized_numpy(quaternions: np.ndarray) -> np.ndarray:
    """Vectorized SQUAD tangent computation for all keyframes (no loop)."""
    # np.roll + boundary fix is cleaner than concatenate
    q_prev = np.roll(quaternions, 1, axis=0)
    q_prev[0] = quaternions[0]  # Replicate first
    q_next = np.roll(quaternions, -1, axis=0)
    q_next[-1] = quaternions[-1]  # Replicate last
    
    return _squad_tangent(q_prev, quaternions, q_next)


def _squad_tangents_vectorized_torch(quaternions: torch.Tensor) -> torch.Tensor:
    """Vectorized SQUAD tangent computation for all keyframes (no loop)."""
    # torch.roll + boundary fix
    q_prev = torch.roll(quaternions, 1, dims=0)
    q_prev[0] = quaternions[0]  # Replicate first
    q_next = torch.roll(quaternions, -1, dims=0)
    q_next[-1] = quaternions[-1]  # Replicate last
    
    return _squad_tangent(q_prev, quaternions, q_next)


def _squad_numpy(quaternions: np.ndarray, times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    n = quaternions.shape[0]
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)

    # Vectorized tangent computation (no loop)
    tangents = _squad_tangents_vectorized_numpy(quaternions)

    indices = np.clip(np.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
    alpha = np.clip((query_times - times[indices]) / (times[indices + 1] - times[indices] + EPS), 0, 1)

    q0, q1 = quaternions[indices], quaternions[indices + 1]
    s0, s1 = tangents[indices], tangents[indices + 1]

    slerp_q = quaternion_slerp(q0, q1, alpha)
    slerp_s = quaternion_slerp(s0, s1, alpha)
    return quaternion_slerp(slerp_q, slerp_s, 2 * alpha * (1 - alpha))


def _squad_torch(quaternions: torch.Tensor, times, query_times) -> torch.Tensor:
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=quaternions.dtype, device=quaternions.device)
    if not isinstance(query_times, torch.Tensor):
        query_times = torch.tensor(query_times, dtype=quaternions.dtype, device=quaternions.device)

    n = quaternions.shape[0]
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    # Vectorized tangent computation (no loop)
    tangents = _squad_tangents_vectorized_torch(quaternions)

    indices = torch.clamp(torch.searchsorted(times, query_times, side="right") - 1, 0, n - 2)
    alpha = torch.clamp((query_times - times[indices]) / (times[indices + 1] - times[indices] + EPS), 0, 1)

    q0, q1 = quaternions[indices], quaternions[indices + 1]
    s0, s1 = tangents[indices], tangents[indices + 1]

    slerp_q = quaternion_slerp(q0, q1, alpha)
    slerp_s = quaternion_slerp(s0, s1, alpha)
    return quaternion_slerp(slerp_q, slerp_s, 2 * alpha * (1 - alpha))


# =============================================================================
# Internal: Minimum Jerk Implementation
# =============================================================================


def _compute_min_jerk_coeffs(p0, p1, v0, v1, a0, a1, T):
    """Compute 5th-order polynomial coefficients."""
    T2, T3, T4, T5 = T*T, T*T*T, T*T*T*T, T*T*T*T*T
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    c3 = (20*(p1 - p0) - (8*v1 + 12*v0)*T - (3*a0 - a1)*T2) / (2*T3 + EPS)
    c4 = (30*(p0 - p1) + (14*v1 + 16*v0)*T + (3*a0 - 2*a1)*T2) / (2*T4 + EPS)
    c5 = (12*(p1 - p0) - 6*(v1 + v0)*T - (a0 - a1)*T2) / (2*T5 + EPS)
    return c0, c1, c2, c3, c4, c5


def _min_jerk_numpy(start, end, t, duration, v0, v1, a0, a1) -> np.ndarray:
    start, end, t = np.asarray(start), np.asarray(end), np.asarray(t)
    v0 = np.zeros_like(start) if v0 is None else np.asarray(v0)
    v1 = np.zeros_like(end) if v1 is None else np.asarray(v1)
    a0 = np.zeros_like(start) if a0 is None else np.asarray(a0)
    a1 = np.zeros_like(end) if a1 is None else np.asarray(a1)

    c0, c1, c2, c3, c4, c5 = _compute_min_jerk_coeffs(start, end, v0, v1, a0, a1, duration)
    t_exp = t if t.ndim == 0 else t.reshape((-1,) + (1,) * start.ndim)
    return c0 + c1*t_exp + c2*t_exp**2 + c3*t_exp**3 + c4*t_exp**4 + c5*t_exp**5


def _min_jerk_torch(start, end, t, duration, v0, v1, a0, a1) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=start.dtype, device=start.device)
    v0 = torch.zeros_like(start) if v0 is None else v0
    v1 = torch.zeros_like(end) if v1 is None else v1
    a0 = torch.zeros_like(start) if a0 is None else a0
    a1 = torch.zeros_like(end) if a1 is None else a1

    c0, c1, c2, c3, c4, c5 = _compute_min_jerk_coeffs(start, end, v0, v1, a0, a1, duration)
    t_exp = t if t.ndim == 0 else t.view((-1,) + (1,) * start.ndim)
    return c0 + c1*t_exp + c2*t_exp**2 + c3*t_exp**3 + c4*t_exp**4 + c5*t_exp**5


def _min_jerk_deriv_numpy(start, end, t, duration, v0, v1, a0, a1, order) -> np.ndarray:
    start, end, t = np.asarray(start), np.asarray(end), np.asarray(t)
    v0 = np.zeros_like(start) if v0 is None else np.asarray(v0)
    v1 = np.zeros_like(end) if v1 is None else np.asarray(v1)
    a0 = np.zeros_like(start) if a0 is None else np.asarray(a0)
    a1 = np.zeros_like(end) if a1 is None else np.asarray(a1)

    _, c1, c2, c3, c4, c5 = _compute_min_jerk_coeffs(start, end, v0, v1, a0, a1, duration)
    t_exp = t if t.ndim == 0 else t.reshape((-1,) + (1,) * start.ndim)

    if order == 1:
        return c1 + 2*c2*t_exp + 3*c3*t_exp**2 + 4*c4*t_exp**3 + 5*c5*t_exp**4
    return 2*c2 + 6*c3*t_exp + 12*c4*t_exp**2 + 20*c5*t_exp**3


def _min_jerk_deriv_torch(start, end, t, duration, v0, v1, a0, a1, order) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=start.dtype, device=start.device)
    v0 = torch.zeros_like(start) if v0 is None else v0
    v1 = torch.zeros_like(end) if v1 is None else v1
    a0 = torch.zeros_like(start) if a0 is None else a0
    a1 = torch.zeros_like(end) if a1 is None else a1

    _, c1, c2, c3, c4, c5 = _compute_min_jerk_coeffs(start, end, v0, v1, a0, a1, duration)
    t_exp = t if t.ndim == 0 else t.view((-1,) + (1,) * start.ndim)

    if order == 1:
        return c1 + 2*c2*t_exp + 3*c3*t_exp**2 + 4*c4*t_exp**3 + 5*c5*t_exp**4
    return 2*c2 + 6*c3*t_exp + 12*c4*t_exp**2 + 20*c5*t_exp**3


# =============================================================================
# Internal: Cubic Spline Implementation
# =============================================================================


def _solve_tridiagonal(a, b, c, d):
    """Solve tridiagonal system using Thomas algorithm."""
    backend = get_backend(a)
    n = b.shape[0]

    if backend == "numpy":
        c_p, d_p = np.zeros_like(c), np.zeros_like(d)
        c_p[0] = c[0] / (b[0] + EPS)
        d_p[0] = d[0] / (b[0] + EPS)
        for i in range(1, n):
            denom = b[i] - a[i] * c_p[i-1] + EPS
            c_p[i] = c[i] / denom if i < n-1 else 0
            d_p[i] = (d[i] - a[i] * d_p[i-1]) / denom
        x = np.zeros_like(d)
        x[-1] = d_p[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_p[i] - c_p[i] * x[i+1]
        return x

    c_p, d_p = torch.zeros_like(c), torch.zeros_like(d)
    c_p[0] = c[0] / (b[0] + EPS)
    d_p[0] = d[0] / (b[0] + EPS)
    for i in range(1, n):
        denom = b[i] - a[i] * c_p[i-1] + EPS
        if i < n-1:
            c_p[i] = c[i] / denom
        d_p[i] = (d[i] - a[i] * d_p[i-1]) / denom
    x = torch.zeros_like(d)
    x[-1] = d_p[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_p[i] - c_p[i] * x[i+1]
    return x


def _spline_coeffs_numpy(points, times, boundary, start_deriv, end_deriv):
    points = np.asarray(points)
    times = np.asarray(times, dtype=points.dtype)
    n = points.shape[0]
    d_dim = points.shape[1] if points.ndim > 1 else 1

    if n < 2:
        raise ValueError("Need at least 2 points")

    h = np.diff(times)
    a = points[:-1].copy()

    if n == 2:
        b = (points[1] - points[0]) / (h[0] + EPS)
        c = np.zeros_like(a)
        d = np.zeros_like(a)
        return a.reshape(1, -1), b.reshape(1, -1), c.reshape(1, -1), d.reshape(1, -1)

    if points.ndim == 1:
        points = points.reshape(-1, 1)
        a = a.reshape(-1, 1)

    delta = np.diff(points, axis=0) / (h[:, np.newaxis] + EPS)
    c_all = np.zeros((n, d_dim), dtype=points.dtype)

    for dim in range(d_dim):
        diag_a, diag_b, diag_c, rhs = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        if boundary == "natural":
            diag_b[0] = diag_b[-1] = 1.0
        elif boundary == "clamped":
            diag_b[0], diag_c[0] = 2*h[0], h[0]
            s_d = np.zeros(d_dim) if start_deriv is None else np.asarray(start_deriv)
            rhs[0] = 3 * (delta[0, dim] - s_d[dim])
            diag_a[-1], diag_b[-1] = h[-1], 2*h[-1]
            e_d = np.zeros(d_dim) if end_deriv is None else np.asarray(end_deriv)
            rhs[-1] = 3 * (e_d[dim] - delta[-1, dim])
        else:  # not-a-knot
            diag_b[0], diag_c[0] = h[1], -(h[0] + h[1])
            diag_a[-1], diag_b[-1] = -(h[-2] + h[-1]), h[-2]

        for i in range(1, n-1):
            diag_a[i], diag_b[i], diag_c[i] = h[i-1], 2*(h[i-1]+h[i]), h[i]
            rhs[i] = 3 * (delta[i, dim] - delta[i-1, dim])

        c_all[:, dim] = _solve_tridiagonal(diag_a, diag_b, diag_c, rhs)

    c = c_all[:-1]
    c_next = c_all[1:]
    b = delta - h[:, np.newaxis] * (2*c + c_next) / 3
    d = (c_next - c) / (3 * h[:, np.newaxis] + EPS)

    if d_dim == 1:
        return a.reshape(-1), b.reshape(-1), c.reshape(-1), d.reshape(-1)
    return a, b, c, d


def _spline_coeffs_torch(points, times, boundary, start_deriv, end_deriv):
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=points.dtype, device=points.device)

    n = points.shape[0]
    d_dim = points.shape[1] if points.ndim > 1 else 1

    if n < 2:
        raise ValueError("Need at least 2 points")

    h = times[1:] - times[:-1]
    a = points[:-1].clone()

    if n == 2:
        b = (points[1] - points[0]) / (h[0] + EPS)
        c = torch.zeros_like(a)
        d = torch.zeros_like(a)
        return a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), d.unsqueeze(0)

    if points.ndim == 1:
        points = points.unsqueeze(-1)
        a = a.unsqueeze(-1)

    delta = (points[1:] - points[:-1]) / (h.unsqueeze(-1) + EPS)
    c_all = torch.zeros((n, d_dim), dtype=points.dtype, device=points.device)

    for dim in range(d_dim):
        diag_a = torch.zeros(n, dtype=points.dtype, device=points.device)
        diag_b = torch.zeros(n, dtype=points.dtype, device=points.device)
        diag_c = torch.zeros(n, dtype=points.dtype, device=points.device)
        rhs = torch.zeros(n, dtype=points.dtype, device=points.device)

        if boundary == "natural":
            diag_b[0] = diag_b[-1] = 1.0
        elif boundary == "clamped":
            diag_b[0], diag_c[0] = 2*h[0], h[0]
            s_d = torch.zeros(d_dim, dtype=points.dtype, device=points.device) if start_deriv is None else start_deriv
            rhs[0] = 3 * (delta[0, dim] - s_d[dim])
            diag_a[-1], diag_b[-1] = h[-1], 2*h[-1]
            e_d = torch.zeros(d_dim, dtype=points.dtype, device=points.device) if end_deriv is None else end_deriv
            rhs[-1] = 3 * (e_d[dim] - delta[-1, dim])
        else:  # not-a-knot
            diag_b[0], diag_c[0] = h[1], -(h[0] + h[1])
            diag_a[-1], diag_b[-1] = -(h[-2] + h[-1]), h[-2]

        for i in range(1, n-1):
            diag_a[i], diag_b[i], diag_c[i] = h[i-1], 2*(h[i-1]+h[i]), h[i]
            rhs[i] = 3 * (delta[i, dim] - delta[i-1, dim])

        c_all[:, dim] = _solve_tridiagonal(diag_a, diag_b, diag_c, rhs)

    c = c_all[:-1]
    c_next = c_all[1:]
    b = delta - h.unsqueeze(-1) * (2*c + c_next) / 3
    d = (c_next - c) / (3 * h.unsqueeze(-1) + EPS)

    if d_dim == 1:
        return a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)
    return a, b, c, d


def _evaluate_spline(spline: SplineCoefficients, query_times: ArrayLike) -> ArrayLike:
    """Evaluate spline at query times."""
    if spline.backend == "numpy":
        qt = np.asarray(query_times)
        times = np.asarray(spline.times)
        idx = np.clip(np.searchsorted(times, qt, side="right") - 1, 0, spline.n_segments - 1)
        t_loc = qt - times[idx]
        if spline.a.ndim > 1:
            t_loc = t_loc[:, np.newaxis]
        return spline.a[idx] + spline.b[idx]*t_loc + spline.c[idx]*t_loc**2 + spline.d[idx]*t_loc**3

    if not isinstance(query_times, torch.Tensor):
        query_times = torch.tensor(query_times, dtype=spline.a.dtype, device=spline.a.device)
    times = spline.times if isinstance(spline.times, torch.Tensor) else torch.tensor(spline.times, dtype=spline.a.dtype, device=spline.a.device)
    idx = torch.clamp(torch.searchsorted(times, query_times, side="right") - 1, 0, spline.n_segments - 1)
    t_loc = query_times - times[idx]
    if spline.a.ndim > 1:
        t_loc = t_loc.unsqueeze(-1)
    return spline.a[idx] + spline.b[idx]*t_loc + spline.c[idx]*t_loc**2 + spline.d[idx]*t_loc**3


def _evaluate_spline_derivative(spline: SplineCoefficients, query_times: ArrayLike, order: int) -> ArrayLike:
    """Evaluate spline derivative at query times."""
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3, got {order}")

    if spline.backend == "numpy":
        qt = np.asarray(query_times)
        times = np.asarray(spline.times)
        idx = np.clip(np.searchsorted(times, qt, side="right") - 1, 0, spline.n_segments - 1)
        t_loc = qt - times[idx]
        if spline.a.ndim > 1:
            t_loc = t_loc[:, np.newaxis]
        if order == 1:
            return spline.b[idx] + 2*spline.c[idx]*t_loc + 3*spline.d[idx]*t_loc**2
        elif order == 2:
            return 2*spline.c[idx] + 6*spline.d[idx]*t_loc
        return 6*spline.d[idx]

    if not isinstance(query_times, torch.Tensor):
        query_times = torch.tensor(query_times, dtype=spline.a.dtype, device=spline.a.device)
    times = spline.times if isinstance(spline.times, torch.Tensor) else torch.tensor(spline.times, dtype=spline.a.dtype, device=spline.a.device)
    idx = torch.clamp(torch.searchsorted(times, query_times, side="right") - 1, 0, spline.n_segments - 1)
    t_loc = query_times - times[idx]
    if spline.a.ndim > 1:
        t_loc = t_loc.unsqueeze(-1)
    if order == 1:
        return spline.b[idx] + 2*spline.c[idx]*t_loc + 3*spline.d[idx]*t_loc**2
    elif order == 2:
        return 2*spline.c[idx] + 6*spline.d[idx]*t_loc
    return 6*spline.d[idx]
