"""
Unified transform utilities supporting both NumPy and PyTorch backends.

API Styles
----------
This library provides two API styles for rotation conversions:

1. Direct Functions (recommended for fixed conversions):
   - Type-safe, zero runtime overhead, clear signatures
   - Examples: quaternion_to_matrix(), euler_to_matrix(), matrix_to_rotvec()

2. Generic Functions (recommended for dynamic/configurable conversions):
   - Flexible, representation determined at runtime
   - Examples: convert_rotation(), rotation_to_matrix(), matrix_to_rotation()

For full poses (rotation + translation), use the Transform class.

Usage Examples
--------------
Direct rotation conversion (fixed path):
    matrix = quaternion_to_matrix(quat)
    euler = matrix_to_euler(matrix, seq="ZYX")

Generic rotation conversion (dynamic path):
    matrix = convert_rotation(rotation, from_rep=input_format, to_rep="matrix")

Transform class (for poses):
    tf = Transform.from_rep(
        np.array([x, y, z, roll, pitch, yaw]),
        from_rep="euler",
        seq="ZYX",
    )
    quat_rep = tf.to_rep("quat")  # [x, y, z, qx, qy, qz, qw]

Batch Operations:
    # All functions support arbitrary batch dimensions
    batch_quats = torch.randn(100, 4)  # (batch, 4)
    batch_matrices = quaternion_to_matrix(batch_quats)  # (batch, 3, 3)

Differentiable Operations:
    # Most PyTorch operations support gradients
    # Exception: matrix_to_euler (use matrix_to_euler_differentiable instead)
    pred_rot6d = model(input)
    pred_matrix = rotation_6d_to_matrix(pred_rot6d)  # gradient flows through
    loss = geodesic_distance(pred_matrix, target_matrix, reduce=False).mean()
    loss.backward()

Conventions
-----------
- Quaternion: xyzw (matches SciPy/ROS convention)
- Euler: default seq="ZYX", can be overridden
- Rotation matrix: (..., 3, 3)
- Transform matrix: (..., 4, 4), rotation in [:3, :3], translation in [:3, 3]

Numerical Stability Notes
-------------------------
- 180-degree rotations: Some functions may have reduced numerical stability
  near 180-degree rotations (w ≈ 0 for quaternions, gimbal lock for euler)
- For training: Prefer rotation_6d representation (continuous, no singularities)
- For inference: quaternion or matrix representations are typically fine
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple, TypeVar, Union, overload

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as ScipyRotation
from enum import Enum


# =============================================================================
# Type Definitions & Constants
# =============================================================================
# Rotation Representation Enum
class RotationRepr(str, Enum):
    EULER = "euler"  
    QUAT = "quat"
    MATRIX = "matrix"   
    ROTATION_6D = "rotation_6d"
    ROT_VEC = "rot_vec"

# Generic type variable for preserving input type in return
T = TypeVar("T", np.ndarray, torch.Tensor)
ArrayLike = Union[np.ndarray, torch.Tensor]
Backend = Literal["numpy", "torch"]

# Numerical constants - use these instead of magic numbers
EPS = 1e-8  # General epsilon for division safety
SMALL_ANGLE_THRESHOLD = 1e-6  # Below this, use Taylor approximations


# Most robot states use RPY order [roll, pitch, yaw], 
# but seq like "ZYX" expects [yaw, pitch, roll]
_EULER_RPY_TO_SEQ_MAPPING = {
    "ZYX": (2, 1, 0),  # rpy -> ypr
    "XYZ": (0, 1, 2),  # rpy -> rpy
    "YZX": (1, 2, 0),
    "ZXY": (2, 0, 1),
    "XZY": (0, 2, 1),
    "YXZ": (1, 0, 2),
}

_EULER_SEQ_TO_RPY_MAPPING = {
    "ZYX": (2, 1, 0),  # ypr -> rpy
    "XYZ": (0, 1, 2),  # rpy -> rpy
    "YZX": (2, 0, 1),
    "ZXY": (1, 2, 0),
    "XZY": (0, 2, 1),
    "YXZ": (1, 0, 2),
}


# =============================================================================
# Backend Detection & Utilities
# =============================================================================


def _get_backend(x: ArrayLike) -> Backend:
    """Determine backend from input type."""
    return "torch" if isinstance(x, torch.Tensor) else "numpy"


def _to_backend(
    x: ArrayLike,
    backend: Backend,
    dtype=None,
    device=None,
) -> ArrayLike:
    """Convert array to specified backend."""
    if backend == "torch":
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device) if dtype or device else x
        return torch.as_tensor(x, dtype=dtype, device=device)
    return np.asarray(x, dtype=dtype)


# =============================================================================
# Cached Index Tensors (Performance Optimization)
# =============================================================================


@functools.lru_cache(maxsize=32)
def _cached_indices(indices: Tuple[int, ...], device: str) -> torch.Tensor:
    """Cache index tensors to avoid repeated allocations."""
    torch_device = torch.device(device)
    return torch.tensor(indices, dtype=torch.long, device=torch_device)


# =============================================================================
# Backend-Agnostic Operations (Simple if/else, no Protocol overhead)
# =============================================================================


def _normalize(x: ArrayLike, dim: int = -1, eps: float = EPS) -> ArrayLike:
    """Normalize vectors along specified dimension."""
    if isinstance(x, torch.Tensor):
        return F.normalize(x, dim=dim, eps=eps)
    norm = np.linalg.norm(x, axis=dim, keepdims=True)
    return x / np.maximum(norm, eps)


def _cross(a: ArrayLike, b: ArrayLike, dim: int = -1) -> ArrayLike:
    """Cross product along specified dimension."""
    if isinstance(a, torch.Tensor):
        return torch.cross(a, b, dim=dim)
    return np.cross(a, b, axis=dim)


def _dot_keepdim(a: ArrayLike, b: ArrayLike, dim: int = -1) -> ArrayLike:
    """Dot product keeping dimension."""
    if isinstance(a, torch.Tensor):
        return (a * b).sum(dim, keepdim=True)
    return np.sum(a * b, axis=dim, keepdims=True)


def _cat(arrays: list, dim: int = -1) -> ArrayLike:
    """Concatenate arrays along dimension."""
    if isinstance(arrays[0], torch.Tensor):
        return torch.cat(arrays, dim=dim)
    return np.concatenate(arrays, axis=dim)


def _stack(arrays: list, dim: int = -1) -> ArrayLike:
    """Stack arrays along new dimension."""
    if isinstance(arrays[0], torch.Tensor):
        return torch.stack(arrays, dim=dim)
    return np.stack(arrays, axis=dim)


def _take_indices(x: ArrayLike, indices: Tuple[int, ...], dim: int = -1) -> ArrayLike:
    """Index selection with cached tensors for PyTorch."""
    if isinstance(x, torch.Tensor):
        idx_tensor = _cached_indices(indices, str(x.device))
        return x.index_select(dim, idx_tensor)
    return x[..., list(indices)] if dim == -1 else np.take(x, list(indices), axis=dim)


def _transpose_last_two(x: ArrayLike) -> ArrayLike:
    """Transpose last two dimensions."""
    if isinstance(x, torch.Tensor):
        return x.transpose(-1, -2)
    return np.swapaxes(x, -1, -2)


def _matmul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Matrix multiplication."""
    if isinstance(a, torch.Tensor):
        return torch.matmul(a, b)
    return np.matmul(a, b)


def _eye(n: int, backend: Backend, dtype=None, device=None) -> ArrayLike:
    """Identity matrix."""
    if backend == "torch":
        return torch.eye(n, dtype=dtype, device=device)
    return np.eye(n, dtype=dtype or np.float64)


def _zeros(shape: Tuple, backend: Backend, dtype=None, device=None) -> ArrayLike:
    """Zero tensor/array."""
    if backend == "torch":
        return torch.zeros(shape, dtype=dtype, device=device)
    return np.zeros(shape, dtype=dtype or np.float64)


# =============================================================================
# 6D Rotation Representation (Consistent Gram-Schmidt)
# =============================================================================


@overload
def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray: ...
@overload
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor: ...


def matrix_to_rotation_6d(matrix: ArrayLike) -> ArrayLike:
    """
    Convert rotation matrix to 6D rotation representation.
    
    The 6D representation consists of the first two rows of the rotation matrix,
    flattened as [row0, row1].
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
    
    Returns:
        6D rotation (..., 6)
    """
    row0 = matrix[..., 0, :]
    row1 = matrix[..., 1, :]
    return _cat([row0, row1], dim=-1)


@overload
def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray: ...
@overload
def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor: ...


def rotation_6d_to_matrix(rot_6d: ArrayLike) -> ArrayLike:
    """
    Convert 6D rotation to rotation matrix using Gram-Schmidt orthogonalization.
    
    This implementation is consistent across NumPy and PyTorch backends.
    
    Args:
        rot_6d: 6D rotation representation (..., 6) as [row0, row1]
    
    Returns:
        Rotation matrix (..., 3, 3)
    
    Raises:
        ValueError: If input shape is invalid
    """
    if rot_6d.shape[-1] != 6:
        raise ValueError(f"6D rotation must have shape (..., 6), got {rot_6d.shape}")
    
    a1 = rot_6d[..., :3]  # First row
    a2 = rot_6d[..., 3:6]  # Second row
    
    # Gram-Schmidt orthogonalization (row-based, matching transform_util.py)
    # row1 = normalize(a1)
    # row3 = normalize(row1 × a2)
    # row2 = row3 × row1
    b1 = _normalize(a1, dim=-1)
    b3 = _cross(b1, a2, dim=-1)
    b3 = _normalize(b3, dim=-1)
    b2 = _cross(b3, b1, dim=-1)
    
    # Stack as rows to form rotation matrix
    return _stack([b1, b2, b3], dim=-2)


# =============================================================================
# Quaternion Operations (xyzw convention throughout)
# =============================================================================


@overload
def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor: ...


def quaternion_to_matrix(quat: ArrayLike) -> ArrayLike:
    """
    Convert quaternion (xyzw) to rotation matrix.
    
    Args:
        quat: Quaternion in xyzw format (..., 4)
    
    Returns:
        Rotation matrix (..., 3, 3)
    
    Raises:
        ValueError: If quaternion shape is invalid
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have shape (..., 4), got {quat.shape}")
    
    backend = _get_backend(quat)
    
    if backend == "numpy":
        # SciPy only supports (4,) or (N, 4), so we need to handle arbitrary batch dims
        batch_shape = quat.shape[:-1]
        quat_flat = quat.reshape(-1, 4)
        matrix_flat = ScipyRotation.from_quat(quat_flat).as_matrix()
        return matrix_flat.reshape(*batch_shape, 3, 3)
    
    # PyTorch implementation
    x, y, z, w = torch.unbind(quat, dim=-1)
    
    # Compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1)
    
    return matrix.reshape(quat.shape[:-1] + (3, 3))


@overload
def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray: ...
@overload
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor: ...


def matrix_to_quaternion(matrix: ArrayLike) -> ArrayLike:
    """
    Convert rotation matrix to quaternion (xyzw).
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
    
    Returns:
        Quaternion in xyzw format (..., 4)
    """
    backend = _get_backend(matrix)
    
    if backend == "numpy":
        # SciPy only supports (3, 3) or (N, 3, 3), handle arbitrary batch dims
        batch_shape = matrix.shape[:-2]
        matrix_flat = matrix.reshape(-1, 3, 3)
        quat_flat = ScipyRotation.from_matrix(matrix_flat).as_quat()
        return quat_flat.reshape(*batch_shape, 4)
    
    # PyTorch: Shepperd's method
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    
    # Four possible solutions based on largest diagonal element
    quat = torch.zeros(m.shape[0], 4, dtype=matrix.dtype, device=matrix.device)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        quat[mask1, 3] = 0.25 * s
        quat[mask1, 0] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
        quat[mask1, 1] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s
        quat[mask1, 2] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s
    
    # Case 2: m[0,0] is largest
    mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2]) * 2
        quat[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s
        quat[mask2, 0] = 0.25 * s
        quat[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
        quat[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s
    
    # Case 3: m[1,1] is largest
    mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2]) * 2
        quat[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s
        quat[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s
        quat[mask3, 1] = 0.25 * s
        quat[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s
    
    # Case 4: m[2,2] is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1]) * 2
        quat[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s
        quat[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s
        quat[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s
        quat[mask4, 2] = 0.25 * s
    
    # Ensure positive w (canonical form)
    quat = torch.where(quat[:, 3:4] < 0, -quat, quat)
    
    return quat.reshape(batch_shape + (4,))


@overload
def quaternion_conjugate(q: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor: ...


def quaternion_conjugate(q: ArrayLike) -> ArrayLike:
    """
    Compute quaternion conjugate.
    
    For unit quaternions, conjugate equals inverse.
    
    Args:
        q: Quaternion(s) in xyzw format (..., 4)
    
    Returns:
        Conjugate quaternion(s) (..., 4)
    """
    backend = _get_backend(q)
    
    if backend == "numpy":
        result = q.copy()
        result[..., :3] = -result[..., :3]
        return result
    
    # PyTorch
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


@overload
def quaternion_inverse(q: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_inverse(q: torch.Tensor) -> torch.Tensor: ...


def quaternion_inverse(q: ArrayLike) -> ArrayLike:
    """
    Compute quaternion inverse.
    
    For unit quaternions, this is equivalent to conjugate.
    For non-unit quaternions, includes normalization.
    
    Args:
        q: Quaternion(s) in xyzw format (..., 4)
    
    Returns:
        Inverse quaternion(s) (..., 4)
    """
    backend = _get_backend(q)
    conj = quaternion_conjugate(q)
    
    if backend == "numpy":
        norm_sq = np.sum(q * q, axis=-1, keepdims=True)
        return conj / norm_sq
    
    # PyTorch
    norm_sq = (q * q).sum(dim=-1, keepdim=True)
    return conj / norm_sq


@overload
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor: ...


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> ArrayLike:
    """
    Multiply two quaternions (Hamilton product).
    
    The result represents the composition of rotations: first q2, then q1.
    Equivalent to: quaternion_to_matrix(q1) @ quaternion_to_matrix(q2)
    
    Args:
        q1: First quaternion(s) in xyzw format (..., 4)
        q2: Second quaternion(s) in xyzw format (..., 4)
    
    Returns:
        Product quaternion(s) in xyzw format (..., 4)
    
    Example:
        >>> # 90 deg around z, then 90 deg around x
        >>> q1 = np.array([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)])  # 90° x
        >>> q2 = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])  # 90° z
        >>> q_result = quaternion_multiply(q1, q2)  # Combined rotation
    """
    backend = _get_backend(q1)
    
    # Extract components (xyzw format)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Hamilton product formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    if backend == "numpy":
        return np.stack([x, y, z, w], axis=-1)
    
    return torch.stack([x, y, z, w], dim=-1)


@overload
def quaternion_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def quaternion_apply(q: ArrayLike, v: ArrayLike) -> ArrayLike:
    """
    Apply quaternion rotation to vector(s).
    
    More efficient than converting to matrix for single vector rotations.
    Formula: v' = q * v * q^(-1), where v is treated as pure quaternion [v, 0]
    
    Args:
        q: Quaternion(s) in xyzw format (..., 4)
        v: Vector(s) to rotate (..., 3)
    
    Returns:
        Rotated vector(s) (..., 3)
    """
    backend = _get_backend(q)
    
    # Extract quaternion components
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    
    if backend == "numpy":
        # Optimized formula: v' = v + 2*w*(qxyz × v) + 2*(qxyz × (qxyz × v))
        # Let t = 2 * (qxyz × v)
        t = 2 * np.cross(qxyz, v, axis=-1)
        return v + qw * t + np.cross(qxyz, t, axis=-1)
    
    # PyTorch
    t = 2 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


# =============================================================================
# Euler Angles
# =============================================================================


def _single_axis_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Create rotation matrix for single axis rotation (PyTorch only)."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    
    if axis == "X":
        row0 = torch.stack([one, zero, zero], dim=-1)
        row1 = torch.stack([zero, cos, -sin], dim=-1)
        row2 = torch.stack([zero, sin, cos], dim=-1)
    elif axis == "Y":
        row0 = torch.stack([cos, zero, sin], dim=-1)
        row1 = torch.stack([zero, one, zero], dim=-1)
        row2 = torch.stack([-sin, zero, cos], dim=-1)
    elif axis == "Z":
        row0 = torch.stack([cos, -sin, zero], dim=-1)
        row1 = torch.stack([sin, cos, zero], dim=-1)
        row2 = torch.stack([zero, zero, one], dim=-1)
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    return torch.stack([row0, row1, row2], dim=-2)


@overload
def euler_to_matrix(
    euler: np.ndarray, seq: str = "ZYX", degrees: bool = False, euler_in_rpy: bool = False
) -> np.ndarray: ...
@overload
def euler_to_matrix(
    euler: torch.Tensor, seq: str = "ZYX", degrees: bool = False, euler_in_rpy: bool = False
) -> torch.Tensor: ...


def euler_to_matrix(
    euler: ArrayLike,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> ArrayLike:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        euler: Euler angles (..., 3)
        seq: Rotation sequence (e.g., "ZYX", "XYZ")
        degrees: If True, angles are in degrees
        euler_in_rpy: If True, input euler is in [roll, pitch, yaw] order and will be
                      reordered to match seq. If False, input is already in seq order.
    
    Returns:
        Rotation matrix (..., 3, 3)
    """
    backend = _get_backend(euler)
    
    # Reorder if euler_in_rpy is True
    if euler_in_rpy and seq in _EULER_RPY_TO_SEQ_MAPPING:
        idx = _EULER_RPY_TO_SEQ_MAPPING[seq]
        euler = _take_indices(euler, idx, dim=-1)
    
    if backend == "numpy":
        # Handle arbitrary batch dimensions
        batch_shape = euler.shape[:-1]
        euler_flat = euler.reshape(-1, 3)
        matrix_flat = ScipyRotation.from_euler(seq, euler_flat, degrees=degrees).as_matrix()
        return matrix_flat.reshape(*batch_shape, 3, 3)
    
    # PyTorch implementation
    if degrees:
        euler = torch.deg2rad(euler)
    
    angles = torch.unbind(euler, dim=-1)
    matrices = [_single_axis_rotation(axis, angle) for axis, angle in zip(seq, angles)]
    
    result = matrices[0]
    for m in matrices[1:]:
        result = torch.matmul(result, m)
    
    return result


@overload
def matrix_to_euler(
    matrix: np.ndarray, seq: str = "ZYX", degrees: bool = False, euler_in_rpy: bool = False
) -> np.ndarray: ...
@overload
def matrix_to_euler(
    matrix: torch.Tensor, seq: str = "ZYX", degrees: bool = False, euler_in_rpy: bool = False
) -> torch.Tensor: ...


def matrix_to_euler(
    matrix: ArrayLike,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> ArrayLike:
    """
    Convert rotation matrix to Euler angles.
    
    Warning:
        PyTorch version does NOT support gradient backpropagation.
        For differentiable rotation representations, use rotation_6d or quaternion.
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
        seq: Rotation sequence (e.g., "ZYX", "XYZ")
        degrees: If True, return angles in degrees
        euler_in_rpy: If True, output will be reordered to [roll, pitch, yaw] order.
                      If False, output is in seq order.
    
    Returns:
        Euler angles (..., 3)
    """
    backend = _get_backend(matrix)
    
    if backend == "numpy":
        # Handle arbitrary batch dimensions
        batch_shape = matrix.shape[:-2]
        matrix_flat = matrix.reshape(-1, 3, 3)
        euler_flat = ScipyRotation.from_matrix(matrix_flat).as_euler(seq, degrees=degrees)
        euler = euler_flat.reshape(*batch_shape, 3)
        
        # Reorder to RPY if requested
        if euler_in_rpy and seq in _EULER_SEQ_TO_RPY_MAPPING:
            idx = list(_EULER_SEQ_TO_RPY_MAPPING[seq])
            euler = euler[..., idx]
        return euler
    
    # PyTorch: convert via quaternion for numerical stability
    quat = matrix_to_quaternion(matrix)
    
    # Use scipy for the actual euler extraction (most robust)
    # This is a pragmatic choice - euler extraction is tricky with gimbal lock
    quat_np = quat.detach().cpu().numpy()
    batch_shape = quat_np.shape[:-1]
    quat_flat = quat_np.reshape(-1, 4)
    euler_flat = ScipyRotation.from_quat(quat_flat).as_euler(seq, degrees=degrees)
    euler_np = euler_flat.reshape(*batch_shape, 3)
    
    # Reorder to RPY if requested
    if euler_in_rpy and seq in _EULER_SEQ_TO_RPY_MAPPING:
        idx = list(_EULER_SEQ_TO_RPY_MAPPING[seq])
        euler_np = euler_np[..., idx]
    
    return torch.as_tensor(euler_np, dtype=matrix.dtype, device=matrix.device)


def _matrix_to_euler_torch_differentiable(
    matrix: torch.Tensor,
    seq: str = "ZYX",
    degrees: bool = False,
) -> torch.Tensor:
    """
    Pure PyTorch differentiable euler angle extraction.
    
    Supports gradient backpropagation but may have numerical issues near gimbal lock.
    Currently supports common sequences: ZYX, XYZ, ZXY, YXZ, YZX, XZY.
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
        seq: Euler sequence
        degrees: If True, return angles in degrees
    
    Returns:
        Euler angles (..., 3)
    """
    # Axis index mapping
    _AXIS_TO_IDX = {"X": 0, "Y": 1, "Z": 2}
    
    i = _AXIS_TO_IDX[seq[0]]
    j = _AXIS_TO_IDX[seq[1]]
    k = _AXIS_TO_IDX[seq[2]]
    
    # Check if Tait-Bryan (all different axes) or proper Euler (first == last)
    is_tait_bryan = (i != k)
    
    if is_tait_bryan:
        # Tait-Bryan angles (e.g., ZYX, XYZ)
        # Sign based on axis ordering
        sign = 1.0 if (j - i) % 3 == 1 else -1.0
        
        # Extract angles
        # For ZYX: R = Rz(a) @ Ry(b) @ Rx(c)
        # angle2 (middle angle) from matrix element
        sin_angle2 = sign * matrix[..., i, k]
        sin_angle2 = torch.clamp(sin_angle2, -1.0, 1.0)
        angle2 = torch.asin(sin_angle2)
        
        # Note: Near gimbal lock (cos(angle2) ≈ 0), angle1 and angle3 become
        # ambiguous. This implementation does not handle gimbal lock specially,
        # which may cause discontinuities in gradients near singularities.
        # For robust training, prefer rotation_6d representation.
        
        # angle1 and angle3 from remaining elements
        if seq == "ZYX":
            angle1 = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
            angle3 = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
        elif seq == "XYZ":
            angle1 = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
            angle3 = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
        elif seq == "YZX":
            angle1 = torch.atan2(-matrix[..., 2, 0], matrix[..., 0, 0])
            angle3 = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
        elif seq == "ZXY":
            angle1 = torch.atan2(-matrix[..., 0, 1], matrix[..., 1, 1])
            angle3 = torch.atan2(-matrix[..., 2, 0], matrix[..., 2, 2])
        elif seq == "XZY":
            angle1 = torch.atan2(matrix[..., 2, 1], matrix[..., 1, 1])
            angle3 = torch.atan2(matrix[..., 0, 2], matrix[..., 0, 0])
        elif seq == "YXZ":
            angle1 = torch.atan2(matrix[..., 0, 2], matrix[..., 2, 2])
            angle3 = torch.atan2(matrix[..., 1, 0], matrix[..., 1, 1])
        else:
            raise ValueError(f"Unsupported Tait-Bryan sequence: {seq}")
        
        euler = torch.stack([angle1, angle2, angle3], dim=-1)
    else:
        # Proper Euler angles (e.g., ZYZ, XYX) - less common
        raise NotImplementedError(
            f"Proper Euler sequence {seq} not implemented. "
            "Use Tait-Bryan sequences (ZYX, XYZ, etc.) for differentiable euler."
        )
    
    if degrees:
        euler = torch.rad2deg(euler)
    
    return euler


@overload
def matrix_to_euler_differentiable(
    matrix: np.ndarray, seq: str = "ZYX", degrees: bool = False
) -> np.ndarray: ...
@overload
def matrix_to_euler_differentiable(
    matrix: torch.Tensor, seq: str = "ZYX", degrees: bool = False
) -> torch.Tensor: ...


def matrix_to_euler_differentiable(
    matrix: ArrayLike,
    seq: str = "ZYX",
    degrees: bool = False,
) -> ArrayLike:
    """
    Differentiable euler angle extraction with gradient support.
    
    This version supports backpropagation for PyTorch tensors, unlike `matrix_to_euler`.
    
    Warning:
        May have numerical issues near gimbal lock singularities.
        For training, consider using rotation_6d or quaternion representations instead.
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
        seq: Euler sequence (ZYX, XYZ, YZX, ZXY, XZY, YXZ)
        degrees: If True, return angles in degrees
    
    Returns:
        Euler angles (..., 3)
    """
    if isinstance(matrix, torch.Tensor):
        return _matrix_to_euler_torch_differentiable(matrix, seq, degrees)
    # NumPy: use scipy (no gradient needed), handle arbitrary batch dims
    batch_shape = matrix.shape[:-2]
    matrix_flat = matrix.reshape(-1, 3, 3)
    euler_flat = ScipyRotation.from_matrix(matrix_flat).as_euler(seq, degrees=degrees)
    return euler_flat.reshape(*batch_shape, 3)


# =============================================================================
# Axis-Angle (Rotation Vector)
# =============================================================================


@overload
def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray: ...
@overload
def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor: ...


def rotvec_to_matrix(rotvec: ArrayLike) -> ArrayLike:
    """
    Convert rotation vector (axis-angle) to rotation matrix.
    
    Args:
        rotvec: Rotation vector (..., 3) where magnitude is angle in radians
    
    Returns:
        Rotation matrix (..., 3, 3)
    """
    backend = _get_backend(rotvec)
    
    if backend == "numpy":
        # Handle arbitrary batch dimensions
        batch_shape = rotvec.shape[:-1]
        rotvec_flat = rotvec.reshape(-1, 3)
        matrix_flat = ScipyRotation.from_rotvec(rotvec_flat).as_matrix()
        return matrix_flat.reshape(*batch_shape, 3, 3)
    
    # PyTorch: Rodrigues formula with proper batch handling
    batch_shape = rotvec.shape[:-1]
    rotvec_flat = rotvec.reshape(-1, 3)
    
    angle = torch.norm(rotvec_flat, dim=-1, keepdim=True)
    axis = rotvec_flat / torch.clamp(angle, min=1e-8)
    
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    
    # Skew-symmetric matrix K
    x, y, z = torch.unbind(axis, dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    
    # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    R = I + sin_a * K + (1 - cos_a) * torch.matmul(K, K)
    
    # Handle small angles (return identity) with proper masking
    small_angle = (angle.squeeze(-1) < SMALL_ANGLE_THRESHOLD)
    if small_angle.any():
        # Use expand to broadcast identity to batch size
        R = torch.where(
            small_angle.unsqueeze(-1).unsqueeze(-1).expand_as(R),
            I.expand_as(R),
            R,
        )
    
    return R.reshape(*batch_shape, 3, 3)


@overload
def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray: ...
@overload
def matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor: ...


def matrix_to_rotvec(matrix: ArrayLike) -> ArrayLike:
    """
    Convert rotation matrix to rotation vector (axis-angle).
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
    
    Returns:
        Rotation vector (..., 3)
    """
    backend = _get_backend(matrix)
    
    if backend == "numpy":
        # Handle arbitrary batch dimensions
        batch_shape = matrix.shape[:-2]
        matrix_flat = matrix.reshape(-1, 3, 3)
        rotvec_flat = ScipyRotation.from_matrix(matrix_flat).as_rotvec()
        return rotvec_flat.reshape(*batch_shape, 3)
    
    # PyTorch: extract via quaternion
    quat = matrix_to_quaternion(matrix)
    return quaternion_to_rotvec(quat)


@overload
def quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_to_rotvec(quat: torch.Tensor) -> torch.Tensor: ...


def quaternion_to_rotvec(quat: ArrayLike) -> ArrayLike:
    """
    Convert quaternion (xyzw) to rotation vector.
    
    Note:
        Uses sign flipping to ensure w >= 0 (canonical quaternion form).
        This may cause gradient discontinuity near w = 0 (180-degree rotations).
    """
    backend = _get_backend(quat)
    
    if backend == "numpy":
        # Handle arbitrary batch dimensions
        batch_shape = quat.shape[:-1]
        quat_flat = quat.reshape(-1, 4)
        rotvec_flat = ScipyRotation.from_quat(quat_flat).as_rotvec()
        return rotvec_flat.reshape(*batch_shape, 3)
    
    # PyTorch: ensure canonical form (w >= 0)
    # Using sign multiplication instead of where for slightly better gradient flow
    sign = torch.sign(quat[..., 3:4])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # Handle w=0 case
    
    xyz = quat[..., :3] * sign
    w = quat[..., 3:4] * sign
    
    norm = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, w)
    
    # Avoid division by zero with smooth approximation
    # For small angles: angle ≈ 2 * norm / w ≈ 2 * norm (since w ≈ 1)
    scale = torch.where(norm > SMALL_ANGLE_THRESHOLD, angle / norm, 2 * torch.ones_like(angle))
    
    return xyz * scale


@overload
def rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray: ...
@overload
def rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor: ...


def rotvec_to_quaternion(rotvec: ArrayLike) -> ArrayLike:
    """
    Convert rotation vector to quaternion (xyzw).
    
    Uses the formula: q = [axis * sin(θ/2), cos(θ/2)] where θ = ||rotvec||.
    """
    backend = _get_backend(rotvec)
    
    if backend == "numpy":
        # Handle arbitrary batch dimensions
        batch_shape = rotvec.shape[:-1]
        rotvec_flat = rotvec.reshape(-1, 3)
        quat_flat = ScipyRotation.from_rotvec(rotvec_flat).as_quat()
        return quat_flat.reshape(*batch_shape, 4)
    
    # PyTorch
    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    half_angle = angle / 2
    
    # For small angles: sin(θ/2) / θ → 1/2 as θ → 0
    # Taylor: sin(θ/2)/θ = 1/2 - θ²/48 + O(θ⁴), so 0.5 is accurate to O(θ²)
    scale = torch.where(
        angle > SMALL_ANGLE_THRESHOLD,
        torch.sin(half_angle) / angle,
        0.5 * torch.ones_like(angle)
    )
    
    xyz = rotvec * scale
    w = torch.cos(half_angle)
    
    return torch.cat([xyz, w], dim=-1)


# =============================================================================
# Rotation Representation Dispatch
# =============================================================================


# Handler type: (rotation, **kwargs) -> rotation_matrix
RotationHandler = Callable[..., ArrayLike]


def _make_to_matrix_handlers() -> Dict[RotationRepr, RotationHandler]:
    """Create handlers for converting representations to matrix."""
    return {
        RotationRepr.MATRIX: lambda r, **kw: r[..., :3, :3],
        RotationRepr.QUAT: lambda r, **kw: quaternion_to_matrix(r),
        RotationRepr.ROTATION_6D: lambda r, **kw: rotation_6d_to_matrix(r),
        RotationRepr.ROT_VEC: lambda r, **kw: rotvec_to_matrix(r),
        RotationRepr.EULER: lambda r, seq="ZYX", degrees=False, euler_in_rpy=False, **kw: euler_to_matrix(
            r, seq, degrees, euler_in_rpy
        ),
    }


def _make_from_matrix_handlers() -> Dict[RotationRepr, RotationHandler]:
    """Create handlers for converting matrix to representations."""
    return {
        RotationRepr.MATRIX: lambda r, **kw: r,
        RotationRepr.QUAT: lambda r, **kw: matrix_to_quaternion(r),
        RotationRepr.ROTATION_6D: lambda r, **kw: matrix_to_rotation_6d(r),
        RotationRepr.ROT_VEC: lambda r, **kw: matrix_to_rotvec(r),
        RotationRepr.EULER: lambda r, seq="ZYX", degrees=False, euler_in_rpy=False, **kw: matrix_to_euler(
            r, seq, degrees, euler_in_rpy
        ),
    }


_TO_MATRIX_HANDLERS = _make_to_matrix_handlers()
_FROM_MATRIX_HANDLERS = _make_from_matrix_handlers()


@overload
def rotation_to_matrix(
    rotation: np.ndarray,
    from_rep: Union[str, RotationRepr],
    *,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> np.ndarray: ...
@overload
def rotation_to_matrix(
    rotation: torch.Tensor,
    from_rep: Union[str, RotationRepr],
    *,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> torch.Tensor: ...


def rotation_to_matrix(
    rotation: ArrayLike,
    from_rep: Union[str, RotationRepr],
    *,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> ArrayLike:
    """
    Convert any rotation representation to rotation matrix.
    
    Args:
        rotation: Rotation in source representation
        from_rep: Source representation ("euler", "quat", "matrix", "rotation_6d", "rot_vec")
        seq: Euler sequence (only used if from_rep is "euler")
        degrees: If True, euler angles are in degrees
        euler_in_rpy: If True and from_rep is euler, input is in [r,p,y] order
    
    Returns:
        Rotation matrix (..., 3, 3)
    """
    from_rep = RotationRepr(from_rep)
    handler = _TO_MATRIX_HANDLERS.get(from_rep)
    if handler is None:
        raise ValueError(f"Unsupported rotation representation: {from_rep}")
    return handler(rotation, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)


@overload
def matrix_to_rotation(
    matrix: np.ndarray,
    to_rep: Union[str, RotationRepr],
    *,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> np.ndarray: ...
@overload
def matrix_to_rotation(
    matrix: torch.Tensor,
    to_rep: Union[str, RotationRepr],
    *,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> torch.Tensor: ...


def matrix_to_rotation(
    matrix: ArrayLike,
    to_rep: Union[str, RotationRepr],
    *,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> ArrayLike:
    """
    Convert rotation matrix to any rotation representation.
    
    Args:
        matrix: Rotation matrix (..., 3, 3)
        to_rep: Target representation ("euler", "quat", "matrix", "rotation_6d", "rot_vec")
        seq: Euler sequence (only used if to_rep is "euler")
        degrees: If True, return euler angles in degrees
        euler_in_rpy: If True and to_rep is euler, output is in [r,p,y] order
    
    Returns:
        Rotation in target representation
    """
    to_rep = RotationRepr(to_rep)
    handler = _FROM_MATRIX_HANDLERS.get(to_rep)
    if handler is None:
        raise ValueError(f"Unsupported rotation representation: {to_rep}")
    return handler(matrix, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)


@overload
def convert_rotation(
    rotation: np.ndarray,
    *,
    from_rep: Union[str, RotationRepr],
    to_rep: Union[str, RotationRepr],
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> np.ndarray: ...
@overload
def convert_rotation(
    rotation: torch.Tensor,
    *,
    from_rep: Union[str, RotationRepr],
    to_rep: Union[str, RotationRepr],
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> torch.Tensor: ...


def convert_rotation(
    rotation: ArrayLike,
    *,
    from_rep: Union[str, RotationRepr],
    to_rep: Union[str, RotationRepr],
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> ArrayLike:
    """
    Convert between rotation representations.
    
    Args:
        rotation: Rotation in source representation
        from_rep: Source representation
        to_rep: Target representation
        seq: Euler sequence
        degrees: If True, euler angles are in degrees
        euler_in_rpy: If True, euler uses [r,p,y] order instead of seq order
    
    Returns:
        Rotation in target representation
    """
    matrix = rotation_to_matrix(rotation, from_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)
    return matrix_to_rotation(matrix, to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)


# =============================================================================
# Transform Class
# =============================================================================


@dataclass
class Transform:
    """
    Rigid body transform supporting both NumPy and PyTorch backends.
    
    Attributes:
        rotation: Rotation matrix (..., 3, 3)
        translation: Translation vector (..., 3)
        extra: Optional extra state like gripper width (..., K). Participates in
               interpolation and distance calculations.
        backend: "numpy" or "torch" (auto-detected from inputs)
    
    Example:
        >>> # With gripper width
        >>> tf = Transform.from_rep(
        ...     np.array([1, 2, 3, 0, 0, 0, 1, 0.04]),  # [x,y,z,quat,gripper]
        ...     from_rep="quat",
        ...     extra_dims=1,
        ... )
        >>> tf.extra  # array([0.04])
        >>> pose = tf.to_rep("euler")  # [x,y,z,euler,gripper]
    """
    
    rotation: ArrayLike
    translation: ArrayLike
    extra: ArrayLike = None  # Optional: gripper width, joint states, etc. (..., K)
    backend: Backend = field(init=False)
    
    def __post_init__(self) -> None:
        """Validate and normalize inputs."""
        # Detect backend from inputs
        rot_backend = _get_backend(self.rotation)
        trans_backend = _get_backend(self.translation)
        extra_backend = _get_backend(self.extra) if self.extra is not None else None
        
        # Use torch if any input is torch
        is_torch = (
            rot_backend == "torch" or 
            trans_backend == "torch" or 
            extra_backend == "torch"
        )
        self.backend = "torch" if is_torch else "numpy"
        
        # Ensure consistent backend
        if self.backend == "torch":
            if not isinstance(self.rotation, torch.Tensor):
                self.rotation = torch.as_tensor(self.rotation)
            if not isinstance(self.translation, torch.Tensor):
                self.translation = torch.as_tensor(
                    self.translation,
                    dtype=self.rotation.dtype,
                    device=self.rotation.device,
                )
            if self.extra is not None and not isinstance(self.extra, torch.Tensor):
                self.extra = torch.as_tensor(
                    self.extra,
                    dtype=self.rotation.dtype,
                    device=self.rotation.device,
                )
    
    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def identity(
        cls,
        backend: Backend = "numpy",
        dtype=None,
        device=None,
        extra_dims: int = 0,
    ) -> "Transform":
        """Create identity transform with optional extra dimensions (initialized to zero)."""
        rot = _eye(3, backend, dtype=dtype, device=device)
        trans = _zeros((3,), backend, dtype=dtype, device=device)
        extra = _zeros((extra_dims,), backend, dtype=dtype, device=device) if extra_dims > 0 else None
        return cls(rotation=rot, translation=trans, extra=extra)
    
    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> "Transform":
        """Create transform from 4x4 homogeneous matrix."""
        return cls(
            rotation=matrix[..., :3, :3],
            translation=matrix[..., :3, 3],
        )
    
    @classmethod
    def from_rep(
        cls,
        tf: ArrayLike,
        *,
        from_rep: Union[str, RotationRepr],
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
        extra_dims: int = 0,
        requires_grad: bool = False,
    ) -> "Transform":
        """
        Create transform from translation + rotation representation + optional extra.
        
        Args:
            tf: [translation (3), rotation (varies by repr), extra (extra_dims)]
            from_rep: Rotation representation
            seq: Euler sequence (if euler)
            degrees: If euler, whether angles are in degrees
            euler_in_rpy: If True and from_rep is euler, input euler is in [r,p,y] order
            extra_dims: Number of extra dimensions at the end (e.g., 1 for gripper width)
            requires_grad: If True, enable gradient tracking (PyTorch only)
        
        Returns:
            Transform instance with optional extra field
        
        Example:
            >>> # [x, y, z, qx, qy, qz, qw, gripper_width]
            >>> tf = Transform.from_rep(
            ...     np.array([1, 2, 3, 0, 0, 0, 1, 0.04]),
            ...     from_rep="quat",
            ...     extra_dims=1,
            ... )
            >>> tf.extra  # array([0.04])
        """
        from_rep = RotationRepr(from_rep)
        
        # Extract extra dimensions if specified
        extra = None
        if extra_dims > 0:
            extra = tf[..., -extra_dims:]
            tf = tf[..., :-extra_dims]
        
        if from_rep == RotationRepr.MATRIX:
            result = cls.from_matrix(tf)
            result.extra = extra
            if requires_grad and result.backend == "torch":
                result.requires_grad_(True)
            return result
        
        translation = tf[..., :3]
        rotation_part = tf[..., 3:]
        rotation = rotation_to_matrix(
            rotation_part, from_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )
        
        result = cls(rotation=rotation, translation=translation, extra=extra)
        if requires_grad and result.backend == "torch":
            result.requires_grad_(True)
        return result
    
    @classmethod
    def from_pos_quat(
        cls,
        position: ArrayLike,
        quaternion: ArrayLike,
        extra: ArrayLike = None,
    ) -> "Transform":
        """
        Create transform from position and quaternion.
        
        Common format for robot poses (e.g., ROS, URDF).
        
        Args:
            position: Translation vector (..., 3)
            quaternion: Quaternion in xyzw format (..., 4)
            extra: Optional extra state like gripper width (..., K)
        
        Returns:
            Transform instance
        
        Example:
            >>> pos = np.array([1.0, 2.0, 3.0])
            >>> quat = np.array([0, 0, 0, 1])  # identity
            >>> gripper = np.array([0.04])
            >>> tf = Transform.from_pos_quat(pos, quat, extra=gripper)
        """
        rotation = quaternion_to_matrix(quaternion)
        return cls(rotation=rotation, translation=position, extra=extra)
    
    @classmethod
    def stack(cls, transforms: List["Transform"], axis: int = 0) -> "Transform":
        """
        Stack multiple transforms into a batched transform.
        
        Args:
            transforms: List of Transform objects
            axis: Axis along which to stack (default: 0)
        
        Returns:
            Batched Transform (extra is stacked if all transforms have extra)
        
        Example:
            >>> tf1 = Transform.identity()
            >>> tf2 = Transform.from_pos_quat([1, 0, 0], [0, 0, 0, 1])
            >>> batched = Transform.stack([tf1, tf2])  # shape: (2, 3, 3) and (2, 3)
        """
        if not transforms:
            raise ValueError("Cannot stack empty list of transforms")
        
        backend = transforms[0].backend
        
        # Check if all transforms have extra
        all_have_extra = all(tf.extra is not None for tf in transforms)
        
        if backend == "numpy":
            rotation = np.stack([tf.rotation for tf in transforms], axis=axis)
            translation = np.stack([tf.translation for tf in transforms], axis=axis)
            extra = None
            if all_have_extra:
                extra = np.stack([tf.extra for tf in transforms], axis=axis)
        else:
            rotation = torch.stack([tf.rotation for tf in transforms], dim=axis)
            translation = torch.stack([tf.translation for tf in transforms], dim=axis)
            extra = None
            if all_have_extra:
                extra = torch.stack([tf.extra for tf in transforms], dim=axis)
        
        return cls(rotation=rotation, translation=translation, extra=extra)
    
    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------
    
    def as_matrix(self) -> ArrayLike:
        """Convert to 4x4 homogeneous transformation matrix."""
        batch_shape = self.rotation.shape[:-2]
        
        if self.backend == "torch":
            matrix = torch.eye(4, dtype=self.rotation.dtype, device=self.rotation.device)
            if batch_shape:
                matrix = matrix.expand(*batch_shape, 4, 4).clone()
            matrix[..., :3, :3] = self.rotation
            matrix[..., :3, 3] = self.translation
            return matrix
        
        matrix = np.zeros((*batch_shape, 4, 4), dtype=self.rotation.dtype)
        matrix[..., :3, :3] = self.rotation
        matrix[..., :3, 3] = self.translation
        matrix[..., 3, 3] = 1.0
        return matrix
    
    def to_rep(
        self,
        to_rep: Union[str, RotationRepr],
        *,
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
        include_extra: bool = True,
    ) -> ArrayLike:
        """
        Convert to [translation, rotation, extra] in specified representation.
        
        Args:
            to_rep: Target rotation representation
            seq: Euler sequence (if euler)
            degrees: If euler, whether to return degrees
            euler_in_rpy: If True and to_rep is euler, output is in [r,p,y] order
            include_extra: If True and extra exists, append extra to output
        
        Returns:
            Array of [translation (3), rotation (varies), extra (K if exists)]
        
        Example:
            >>> tf = Transform.from_rep([1,2,3,0,0,0,1,0.04], from_rep="quat", extra_dims=1)
            >>> tf.to_rep("euler")  # [x, y, z, euler(3), gripper(1)]
        """
        to_rep = RotationRepr(to_rep)
        
        if to_rep == RotationRepr.MATRIX:
            # Note: Matrix representation returns 4x4, extra not concatenated
            return self.as_matrix()
        
        rotation_repr = matrix_to_rotation(
            self.rotation, to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )
        result = _cat([self.translation, rotation_repr], dim=-1)
        
        # Append extra if exists
        if include_extra and self.extra is not None:
            result = _cat([result, self.extra], dim=-1)
        
        return result
    
    # -------------------------------------------------------------------------
    # Transform Operations
    # -------------------------------------------------------------------------
    
    def __matmul__(self, other: "Transform") -> "Transform":
        """
        Compose transforms: self @ other.
        
        The result transforms points as: self.transform(other.transform(point))
        
        Note:
            Extra from `other` is preserved in the result (the "inner" transform's state).
            This is useful for robot control where you apply a world-frame delta to a pose
            and want to keep the pose's gripper state.
        """
        if self.backend != other.backend:
            raise ValueError(
                f"Cannot compose transforms with different backends: "
                f"{self.backend} vs {other.backend}"
            )
        
        rotation = _matmul(self.rotation, other.rotation)
        translation = (
            _matmul(self.rotation, other.translation[..., None]).squeeze(-1)
            + self.translation
        )
        
        # Use other's extra (the "inner" transform's state)
        # This is the common case: delta @ pose -> keep pose's gripper state
        extra = other.extra if other.extra is not None else self.extra
        
        return Transform(rotation=rotation, translation=translation, extra=extra)
    
    def compose(self, other: "Transform") -> "Transform":
        """Alias for @ operator."""
        return self @ other
    
    def inverse(self) -> "Transform":
        """Compute inverse transform (extra is preserved)."""
        rot_inv = _transpose_last_two(self.rotation)
        trans_inv = -_matmul(rot_inv, self.translation[..., None]).squeeze(-1)
        return Transform(rotation=rot_inv, translation=trans_inv, extra=self.extra)
    
    def transform_point(self, point: ArrayLike) -> ArrayLike:
        """Apply transform to point(s)."""
        rotated = _matmul(self.rotation, point[..., None]).squeeze(-1)
        return rotated + self.translation
    
    def transform_vector(self, vector: ArrayLike) -> ArrayLike:
        """Apply rotation only to vector(s) (no translation)."""
        return _matmul(self.rotation, vector[..., None]).squeeze(-1)
    
    def apply_delta(self, delta: "Transform", in_world_frame: bool = True) -> "Transform":
        """
        Apply a delta (incremental) transform.
        
        Useful for robot control where velocities are integrated.
        
        Args:
            delta: Incremental transform to apply
            in_world_frame: If True, delta is in world frame (result = delta @ self)
                           If False, delta is in local frame (result = self @ delta)
        
        Returns:
            Updated Transform
        
        Example:
            >>> current_pose = Transform.identity()
            >>> delta = Transform.from_pos_quat([0.01, 0, 0], [0, 0, 0, 1])
            >>> new_pose = current_pose.apply_delta(delta)  # Moved 1cm in x
        """
        if in_world_frame:
            return delta @ self
        else:
            return self @ delta
    
    def relative_to(self, reference: "Transform") -> "Transform":
        """
        Compute this transform relative to a reference frame.
        
        Returns T such that: reference @ T = self
        Equivalent to: reference.inverse() @ self
        
        Args:
            reference: Reference transform
        
        Returns:
            Transform in reference frame
        """
        return reference.inverse() @ self
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Get batch dimensions."""
        return self.rotation.shape[:-2]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape."""
        return self.translation.shape, self.rotation.shape
    
    @property
    def is_batched(self) -> bool:
        """Check if transform has batch dimensions."""
        return len(self.batch_shape) > 0
    
    @property
    def num_transforms(self) -> int:
        """Get number of transforms (for batched transforms)."""
        if not self.is_batched:
            return 1
        return int(np.prod(self.batch_shape))
    
    @property
    def device(self):
        """Get device (for PyTorch tensors)."""
        if self.backend == "torch":
            return self.rotation.device
        return None
    
    @property
    def dtype(self):
        """Get data type."""
        return self.rotation.dtype
    
    @property
    def extra_dims(self) -> int:
        """Get number of extra dimensions (0 if no extra)."""
        if self.extra is None:
            return 0
        return self.extra.shape[-1] if self.extra.ndim > 0 else 1
    
    @property
    def requires_grad(self) -> bool:
        """Check if gradients are enabled (PyTorch only)."""
        if self.backend == "torch":
            extra_grad = self.extra.requires_grad if self.extra is not None else False
            return self.rotation.requires_grad or self.translation.requires_grad or extra_grad
        return False
    
    def requires_grad_(self, requires_grad: bool = True) -> "Transform":
        """
        Enable/disable gradient tracking in-place (PyTorch only).

        Args:
            requires_grad: Whether to track gradients

        Returns:
            self (for chaining)
        """
        if self.backend != "torch":
            raise ValueError("requires_grad_() is only supported for PyTorch tensors")
        self.rotation.requires_grad_(requires_grad)
        self.translation.requires_grad_(requires_grad)
        if self.extra is not None:
            self.extra.requires_grad_(requires_grad)
        return self
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def to(self, device=None, dtype=None) -> "Transform":
        """
        Move transform to device/dtype (PyTorch only).
        
        Args:
            device: Target device
            dtype: Target dtype
        
        Returns:
            New Transform on target device/dtype
        """
        if self.backend != "torch":
            raise ValueError("to() is only supported for PyTorch tensors")
        
        rot = self.rotation.to(device=device, dtype=dtype)
        trans = self.translation.to(device=device, dtype=dtype)
        extra = self.extra.to(device=device, dtype=dtype) if self.extra is not None else None
        return Transform(rotation=rot, translation=trans, extra=extra)
    
    def detach(self) -> "Transform":
        """Detach from computation graph (PyTorch only)."""
        if self.backend != "torch":
            return self
        return Transform(
            rotation=self.rotation.detach(),
            translation=self.translation.detach(),
            extra=self.extra.detach() if self.extra is not None else None,
        )
    
    def clone(self) -> "Transform":
        """Create a copy of the transform."""
        if self.backend == "torch":
            return Transform(
                rotation=self.rotation.clone(),
                translation=self.translation.clone(),
                extra=self.extra.clone() if self.extra is not None else None,
            )
        return Transform(
            rotation=self.rotation.copy(),
            translation=self.translation.copy(),
            extra=self.extra.copy() if self.extra is not None else None,
        )
    
    def __repr__(self) -> str:
        """String representation."""
        extra_info = ""
        if self.extra is not None:
            extra_shape = self.extra.shape[-1] if self.extra.ndim > 0 else 1
            extra_info = f", extra_dims={extra_shape}"
        return (
            f"Transform(backend={self.backend}, "
            f"batch_shape={self.batch_shape}{extra_info}, "
            f"dtype={self.dtype})"
        )
    
    def __getitem__(self, idx) -> "Transform":
        """Index into batched transforms."""
        return Transform(
            rotation=self.rotation[idx],
            translation=self.translation[idx],
            extra=self.extra[idx] if self.extra is not None else None,
        )
    
    # -------------------------------------------------------------------------
    # Static Conversion Helper
    # -------------------------------------------------------------------------
    
    @staticmethod
    def convert(
        tf: ArrayLike,
        *,
        from_rep: Union[str, RotationRepr],
        to_rep: Union[str, RotationRepr],
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
    ) -> ArrayLike:
        """
        Convert transform between representations without creating Transform instance.
        
        More efficient for one-off conversions.
        """
        transform = Transform.from_rep(
            tf, from_rep=from_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )
        return transform.to_rep(to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)


# =============================================================================
# Interpolation
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
        t: Interpolation parameter(s) in [0, 1]. Can be scalar or array.
           t=0 returns q0, t=1 returns q1.
    
    Returns:
        Interpolated quaternion(s) (..., 4)
    
    Example:
        >>> q0 = np.array([0, 0, 0, 1])  # identity
        >>> q1 = np.array([0, 0, 0.707, 0.707])  # 90 deg around z
        >>> q_mid = quaternion_slerp(q0, q1, 0.5)  # 45 deg around z
    """
    backend = _get_backend(q0)
    
    if backend == "numpy":
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t)
        
        # Vectorized NumPy SLERP implementation
        # Normalize quaternions
        q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
        
        # Compute dot product
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        
        # Take shorter path
        q1 = np.where(dot < 0, -q1, q1)
        dot = np.abs(dot)
        
        # Clamp for numerical stability
        dot = np.clip(dot, -1.0, 1.0)
        
        # Angle between quaternions
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        # Handle t broadcasting: when t is 1D (N,) and q0 is 1D (4,),
        # we want output shape (N, 4), so t should become (N, 1)
        if np.ndim(t) == 0 or (np.ndim(t) == 1 and t.shape[0] == 1):
            t = np.broadcast_to(t, q0.shape[:-1] + (1,))
        elif t.ndim == 1 and q0.ndim == 1:
            t = np.expand_dims(t, axis=-1)  # (N,) -> (N, 1)
        elif t.ndim < q0.ndim:
            t = np.expand_dims(t, axis=-1)
        
        # SLERP formula with small angle fallback
        small_angle = np.abs(sin_theta) < SMALL_ANGLE_THRESHOLD
        
        # Avoid division by zero
        safe_sin_theta = np.where(small_angle, 1.0, sin_theta)
        
        s0 = np.sin((1 - t) * theta) / safe_sin_theta
        s1 = np.sin(t * theta) / safe_sin_theta
        
        # Use linear interpolation for small angles
        s0 = np.where(small_angle, 1 - t, s0)
        s1 = np.where(small_angle, t, s1)
        
        result = s0 * q0 + s1 * q1
        
        # Normalize result
        return result / np.linalg.norm(result, axis=-1, keepdims=True)
    
    # PyTorch implementation
    q0 = q0.clone()
    q1 = q1.clone()
    
    # Convert t to tensor if needed
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)
    
    # Handle broadcasting: when t is 1D (N,) and q0 is 1D (4,),
    # we want output shape (N, 4), so t should become (N, 1)
    if t.ndim == 1 and q0.ndim == 1:
        t = t.unsqueeze(-1)  # (N,) -> (N, 1)
    else:
        while t.ndim < q0.ndim:
            t = t.unsqueeze(-1)
    
    # Normalize quaternions
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    
    # Compute dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    
    # If dot < 0, negate q1 to take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Angle between quaternions
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # Handle small angles (use linear interpolation)
    # For small theta: sin(theta) ≈ theta, so we use lerp
    small_angle = sin_theta.abs() < SMALL_ANGLE_THRESHOLD
    
    # SLERP formula: (sin((1-t)*θ) * q0 + sin(t*θ) * q1) / sin(θ)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    
    # Linear interpolation for small angles
    s0 = torch.where(small_angle, 1 - t, s0)
    s1 = torch.where(small_angle, t, s1)
    
    result = s0 * q0 + s1 * q1
    
    # Normalize result
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
    
    Faster but less accurate than SLERP. Good for small rotations or
    when many interpolations are needed.
    
    Args:
        q0: Start quaternion(s) in xyzw format (..., 4)
        q1: End quaternion(s) in xyzw format (..., 4)
        t: Interpolation parameter(s) in [0, 1]
    
    Returns:
        Interpolated quaternion(s) (..., 4)
    """
    backend = _get_backend(q0)
    
    if backend == "numpy":
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t)
        
        # Handle t broadcasting: when t is 1D (N,) and q0 is 1D (4,),
        # we want output shape (N, 4), so t should become (N, 1)
        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
            t = np.broadcast_to(t, q0.shape[:-1] + (1,))
        elif t.ndim == 1 and q0.ndim == 1:
            t = np.expand_dims(t, axis=-1)  # (N,) -> (N, 1)
        elif t.ndim < q0.ndim:
            t = np.expand_dims(t, axis=-1)
        
        # Take shorter path
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        
        # Linear interpolation
        result = (1 - t) * q0 + t * q1
        
        # Normalize
        return result / np.linalg.norm(result, axis=-1, keepdims=True)
    
    # PyTorch
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)
    
    # Handle broadcasting: when t is 1D (N,) and q0 is 1D (4,),
    # we want output shape (N, 4), so t should become (N, 1)
    if t.ndim == 1 and q0.ndim == 1:
        t = t.unsqueeze(-1)  # (N,) -> (N, 1)
    else:
        while t.ndim < q0.ndim:
            t = t.unsqueeze(-1)
    
    # Take shorter path
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    
    # Linear interpolation
    result = (1 - t) * q0 + t * q1
    
    # Normalize
    return result / torch.norm(result, dim=-1, keepdim=True)


def transform_interpolate(
    tf0: Transform,
    tf1: Transform,
    t: Union[float, ArrayLike],
    rotation_method: Literal["slerp", "nlerp"] = "slerp",
) -> Transform:
    """
    Interpolate between two transforms.
    
    Uses linear interpolation for translation and extra (e.g., gripper width),
    and spherical interpolation for rotation (via quaternions).
    
    Args:
        tf0: Start transform
        tf1: End transform
        t: Interpolation parameter(s) in [0, 1]. t=0 returns tf0, t=1 returns tf1.
        rotation_method: "slerp" (accurate) or "nlerp" (fast)
    
    Returns:
        Interpolated Transform (including interpolated extra if both have extra)
    
    Example:
        >>> tf0 = Transform.from_rep([0,0,0,0,0,0,1,0.08], from_rep="quat", extra_dims=1)
        >>> tf1 = Transform.from_rep([1,0,0,0,0,0,1,0.00], from_rep="quat", extra_dims=1)
        >>> tf_mid = transform_interpolate(tf0, tf1, 0.5)
        >>> tf_mid.extra  # [0.04] - gripper interpolated
    """
    if tf0.backend != tf1.backend:
        raise ValueError(
            f"Cannot interpolate transforms with different backends: "
            f"{tf0.backend} vs {tf1.backend}"
        )
    
    backend = tf0.backend
    
    # Convert t to appropriate type and handle broadcasting
    # When t is array-like (e.g., shape (N,)), we want output shape (N, 3) for translation
    # So t_trans needs shape (N, 1) to broadcast with translation shape (3,)
    if backend == "torch":
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=tf0.rotation.dtype, device=tf0.rotation.device)
        t_trans = t
        # If t is 1D and translation is 1D, add dimension for broadcasting
        if t_trans.ndim == 1 and tf0.translation.ndim == 1:
            t_trans = t_trans.unsqueeze(-1)  # (N,) -> (N, 1)
        else:
            while t_trans.ndim < tf0.translation.ndim:
                t_trans = t_trans.unsqueeze(-1)
    else:
        t = np.asarray(t)
        t_trans = t
        # If t is 1D and translation is 1D, add dimension for broadcasting
        if t_trans.ndim == 1 and tf0.translation.ndim == 1:
            t_trans = np.expand_dims(t_trans, axis=-1)  # (N,) -> (N, 1)
        elif t_trans.ndim < tf0.translation.ndim:
            t_trans = np.expand_dims(t, axis=-1)
    
    # Linear interpolation for translation
    translation = (1 - t_trans) * tf0.translation + t_trans * tf1.translation
    
    # Linear interpolation for extra (e.g., gripper width)
    extra = None
    if tf0.extra is not None and tf1.extra is not None:
        # Use same t_trans broadcasting for extra
        extra = (1 - t_trans) * tf0.extra + t_trans * tf1.extra
    elif tf0.extra is not None:
        extra = tf0.extra
    elif tf1.extra is not None:
        extra = tf1.extra
    
    # Convert rotation matrices to quaternions
    q0 = matrix_to_quaternion(tf0.rotation)
    q1 = matrix_to_quaternion(tf1.rotation)
    
    # Spherical interpolation for rotation
    if rotation_method == "slerp":  
        q_interp = quaternion_slerp(q0, q1, t)
    elif rotation_method == "nlerp":
        q_interp = quaternion_nlerp(q0, q1, t)
    else:
        raise ValueError(f"Unknown rotation_method: {rotation_method}")
    
    # Convert back to rotation matrix
    rotation = quaternion_to_matrix(q_interp)
    
    return Transform(rotation=rotation, translation=translation, extra=extra)


def transform_sequence_interpolate(
    transforms: Transform,
    times: ArrayLike,
    query_times: ArrayLike,
    rotation_method: Literal["slerp", "nlerp"] = "slerp",
    extrapolate: bool = False,
) -> Transform:
    """
    Interpolate a sequence of transforms at query times (vectorized).
    
    Args:
        transforms: Batched Transform with shape (N, ...) containing keyframes
        times: Times corresponding to each transform (N,)
        query_times: Times at which to interpolate (M,)
        rotation_method: "slerp" or "nlerp"
        extrapolate: If True, allow extrapolation beyond time range.
                    If False (default), clamp to boundary transforms.
    
    Returns:
        Transform with batch dimension (M, ...) including interpolated extra
    
    Example:
        >>> # 100 keyframes with gripper: [x,y,z,rot6d(6),gripper(1)] = 10 dims
        >>> traj = torch.randn(100, 10)
        >>> tfs = Transform.from_rep(traj, from_rep="rotation_6d", extra_dims=1)
        >>> times = torch.linspace(0, 1, 100)
        >>> query = torch.linspace(0, 1, 1000)
        >>> result = transform_sequence_interpolate(tfs, times, query)
        >>> result.extra.shape  # (1000, 1) - gripper also interpolated
    """
    # Get number of keyframes from first batch dimension
    n_keyframes = transforms.rotation.shape[0]
    if n_keyframes < 2:
        raise ValueError("Need at least 2 transforms for interpolation")
    
    backend = transforms.backend
    dtype = transforms.rotation.dtype
    device = transforms.device if backend == "torch" else None
    
    # Get rotations and translations directly from batched transform
    all_rot = transforms.rotation  # (N, 3, 3)
    all_trans = transforms.translation  # (N, 3)
    all_extra = transforms.extra  # (N, K) or None
    
    if backend == "numpy":
        times = np.asarray(times)
        query_times = np.asarray(query_times)
        
        if n_keyframes != len(times):
            raise ValueError(f"transforms ({n_keyframes}) and times ({len(times)}) must have same length")
        
        all_quat = matrix_to_quaternion(all_rot)  # (N, 4)
        
        # Vectorized search for all query times at once
        indices = np.searchsorted(times, query_times, side='right') - 1
        indices = np.clip(indices, 0, n_keyframes - 2)  # (M,)
        
        # Get t0, t1 for each query
        t0 = times[indices]  # (M,)
        t1 = times[indices + 1]  # (M,)
        
        # Compute alpha for each query
        alpha = (query_times - t0) / (t1 - t0 + EPS)  # (M,)
        if not extrapolate:
            alpha = np.clip(alpha, 0, 1)
        
        # Gather rotations and translations for interpolation
        q0 = all_quat[indices]  # (M, 4)
        q1 = all_quat[indices + 1]  # (M, 4)
        trans0 = all_trans[indices]  # (M, 3)
        trans1 = all_trans[indices + 1]  # (M, 3)
        
        # Interpolate translations
        alpha_trans = alpha[:, np.newaxis]  # (M, 1)
        translation = (1 - alpha_trans) * trans0 + alpha_trans * trans1  # (M, 3)
        
        # Interpolate extra (e.g., gripper width)
        extra = None
        if all_extra is not None:
            extra0 = all_extra[indices]  # (M, K)
            extra1 = all_extra[indices + 1]  # (M, K)
            extra = (1 - alpha_trans) * extra0 + alpha_trans * extra1  # (M, K)
        
        # Interpolate rotations
        if rotation_method == "slerp":
            q_interp = quaternion_slerp(q0, q1, alpha)
        else:
            q_interp = quaternion_nlerp(q0, q1, alpha)
        
        rotation = quaternion_to_matrix(q_interp)  # (M, 3, 3)
        
    else:
        # PyTorch vectorized implementation
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times, dtype=dtype, device=device)
        if not isinstance(query_times, torch.Tensor):
            query_times = torch.tensor(query_times, dtype=dtype, device=device)
        
        if n_keyframes != len(times):
            raise ValueError(f"transforms ({n_keyframes}) and times ({len(times)}) must have same length")
        
        all_quat = matrix_to_quaternion(all_rot)  # (N, 4)
        
        # Vectorized search
        indices = torch.searchsorted(times, query_times, side='right') - 1
        indices = torch.clamp(indices, 0, n_keyframes - 2)  # (M,)
        
        # Get t0, t1 for each query
        t0 = times[indices]  # (M,)
        t1 = times[indices + 1]  # (M,)
        
        # Compute alpha
        alpha = (query_times - t0) / (t1 - t0 + EPS)  # (M,)
        if not extrapolate:
            alpha = torch.clamp(alpha, 0, 1)
        
        # Gather using advanced indexing
        q0 = all_quat[indices]  # (M, 4)
        q1 = all_quat[indices + 1]  # (M, 4)
        trans0 = all_trans[indices]  # (M, 3)
        trans1 = all_trans[indices + 1]  # (M, 3)
        
        # Interpolate translations
        alpha_trans = alpha.unsqueeze(-1)  # (M, 1)
        translation = (1 - alpha_trans) * trans0 + alpha_trans * trans1  # (M, 3)
        
        # Interpolate extra (e.g., gripper width)
        extra = None
        if all_extra is not None:
            extra0 = all_extra[indices]  # (M, K)
            extra1 = all_extra[indices + 1]  # (M, K)
            extra = (1 - alpha_trans) * extra0 + alpha_trans * extra1  # (M, K)
        
        # Interpolate rotations
        if rotation_method == "slerp":
            q_interp = quaternion_slerp(q0, q1, alpha)
        else:
            q_interp = quaternion_nlerp(q0, q1, alpha)
        
        rotation = quaternion_to_matrix(q_interp)  # (M, 3, 3)
    
    return Transform(rotation=rotation, translation=translation, extra=extra)


# =============================================================================
# SE(3) Lie Group Operations
# =============================================================================


@overload
def se3_log(transform: Transform) -> np.ndarray: ...
@overload
def se3_log(transform: Transform) -> torch.Tensor: ...


def se3_log(transform: Transform) -> ArrayLike:
    """
    Compute SE(3) logarithm (transform to twist).
    
    Maps a rigid transform to its corresponding se(3) Lie algebra element
    (6D twist vector: [angular_velocity, linear_velocity]).
    
    Args:
        transform: Rigid body transform
    
    Returns:
        6D twist vector (..., 6) as [omega (3), v (3)]
        - omega: rotation vector (axis * angle)
        - v: linear velocity component
    
    Note:
        For small rotations, this is approximately [rotation_vector, translation].
        For larger rotations, the translation component is adjusted.
    """
    backend = transform.backend
    
    # Get rotation vector (axis-angle)
    omega = matrix_to_rotvec(transform.rotation)  # (..., 3)
    
    if backend == "numpy":
        angle = np.linalg.norm(omega, axis=-1, keepdims=True)
        
        # For small angles, V^{-1} ≈ I
        small_angle = np.abs(angle) < SMALL_ANGLE_THRESHOLD
        
        # Compute V^{-1} for the translation adjustment
        # V^{-1} = I - 0.5*[omega]_x + (1/angle^2)(1 - angle/(2*tan(angle/2)))*[omega]_x^2
        # For small angles: V^{-1} ≈ I - 0.5*[omega]_x
        
        if np.all(small_angle):
            # Simple case: small rotation
            v = transform.translation.copy()
        else:
            # General case: need to adjust translation
            # Using the approximation v = V^{-1} @ t
            # For implementation simplicity, we use the small angle approximation
            # which is accurate enough for most robotics applications
            axis = omega / (angle + EPS)
            half_angle = angle / 2
            
            # V^{-1} t ≈ t - 0.5 * (omega × t) + correction term
            omega_cross_t = np.cross(omega, transform.translation, axis=-1)
            correction = (1 - angle * np.cos(half_angle) / (2 * np.sin(half_angle) + EPS)) / (angle * angle + EPS)
            omega_cross_omega_cross_t = np.cross(omega, omega_cross_t, axis=-1)
            
            v = np.where(
                small_angle,
                transform.translation,
                transform.translation - 0.5 * omega_cross_t + correction * omega_cross_omega_cross_t
            )
        
        return np.concatenate([omega, v], axis=-1)
    
    # PyTorch implementation
    angle = torch.norm(omega, dim=-1, keepdim=True)
    small_angle = angle.abs() < SMALL_ANGLE_THRESHOLD
    
    # For small angles: v ≈ translation
    # For larger angles: apply V^{-1} correction
    axis = omega / (angle + EPS)
    half_angle = angle / 2
    
    omega_cross_t = torch.cross(omega, transform.translation, dim=-1)
    correction = (1 - angle * torch.cos(half_angle) / (2 * torch.sin(half_angle) + EPS)) / (angle * angle + EPS)
    omega_cross_omega_cross_t = torch.cross(omega, omega_cross_t, dim=-1)
    
    v = torch.where(
        small_angle,
        transform.translation,
        transform.translation - 0.5 * omega_cross_t + correction * omega_cross_omega_cross_t
    )
    
    return torch.cat([omega, v], dim=-1)


@overload
def se3_exp(twist: np.ndarray) -> Transform: ...
@overload
def se3_exp(twist: torch.Tensor) -> Transform: ...


def se3_exp(twist: ArrayLike) -> Transform:
    """
    Compute SE(3) exponential (twist to transform).
    
    Maps a 6D twist (se(3) Lie algebra element) to a rigid transform.
    
    Args:
        twist: 6D twist vector (..., 6) as [omega (3), v (3)]
               - omega: rotation vector (axis * angle)
               - v: linear velocity component
    
    Returns:
        Transform corresponding to the twist
    
    Note:
        This is the inverse of se3_log.
        For small rotations: rotation = exp(omega), translation ≈ v
    """
    backend = _get_backend(twist)
    
    omega = twist[..., :3]
    v = twist[..., 3:6]
    
    # Rotation from rotation vector
    rotation = rotvec_to_matrix(omega)  # (..., 3, 3)
    
    if backend == "numpy":
        angle = np.linalg.norm(omega, axis=-1, keepdims=True)
        small_angle = np.abs(angle) < SMALL_ANGLE_THRESHOLD
        
        if np.all(small_angle):
            # Small rotation: translation ≈ v
            translation = v.copy()
        else:
            # General case: t = V @ v
            # V = I + (1-cos(θ))/θ² [ω]_× + (θ-sin(θ))/θ³ [ω]_×²
            axis = omega / (angle + EPS)
            
            # Compute V @ v using Rodriguez-like formula
            omega_cross_v = np.cross(omega, v, axis=-1)
            omega_cross_omega_cross_v = np.cross(omega, omega_cross_v, axis=-1)
            
            c1 = (1 - np.cos(angle)) / (angle * angle + EPS)
            c2 = (angle - np.sin(angle)) / (angle * angle * angle + EPS)
            
            translation = np.where(
                small_angle,
                v,
                v + c1 * omega_cross_v + c2 * omega_cross_omega_cross_v
            )
    else:
        # PyTorch
        angle = torch.norm(omega, dim=-1, keepdim=True)
        small_angle = angle.abs() < SMALL_ANGLE_THRESHOLD
        
        omega_cross_v = torch.cross(omega, v, dim=-1)
        omega_cross_omega_cross_v = torch.cross(omega, omega_cross_v, dim=-1)
        
        c1 = (1 - torch.cos(angle)) / (angle * angle + EPS)
        c2 = (angle - torch.sin(angle)) / (angle * angle * angle + EPS)
        
        translation = torch.where(
            small_angle,
            v,
            v + c1 * omega_cross_v + c2 * omega_cross_omega_cross_v
        )
    
    return Transform(rotation=rotation, translation=translation)


# =============================================================================
# Additional Utilities
# =============================================================================


def xyz_rotation_6d_to_matrix(xyz_rot_6d: ArrayLike) -> ArrayLike:
    """
    Convert [x, y, z, 6D rotation] to 4x4 homogeneous transformation matrix.
    
    Args:
        xyz_rot_6d: Array of shape (..., 9) containing [x, y, z, rot6d]
    
    Returns:
        Homogeneous matrix (..., 4, 4)
    """
    if xyz_rot_6d.shape[-1] != 9:
        raise ValueError(f"Expected last dimension 9, got {xyz_rot_6d.shape[-1]}")
    
    translation = xyz_rot_6d[..., :3]
    rot_6d = xyz_rot_6d[..., 3:9]
    rotation = rotation_6d_to_matrix(rot_6d)
    
    return Transform(rotation=rotation, translation=translation).as_matrix()


# Overloads for precise return type based on reduce parameter
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
    backend = _get_backend(R1)
    
    # Compute R1^T @ R2
    R_diff = _matmul(_transpose_last_two(R1), R2)
    
    # Trace
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
    backend = _get_backend(t1)
    
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

@overload
def transform_distance(
    tf1: Transform,
    tf2: Transform,
    reduce: Literal[True] = ...,
    rotation_weight: float = ...,
    translation_weight: float = ...,
    extra_weight: float = ...,
    degrees: bool = ...,
) -> Tuple[float, float, float, float]: ...
@overload
def transform_distance(
    tf1: Transform,
    tf2: Transform,
    reduce: Literal[False] = ...,
    rotation_weight: float = ...,
    translation_weight: float = ...,
    extra_weight: float = ...,
    degrees: bool = ...,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: ...


def transform_distance(
    tf1: Transform,
    tf2: Transform,
    reduce: bool = True,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
    extra_weight: float = 1.0,
    degrees: bool = False,
) -> Union[Tuple[float, float, float, float], Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]]:
    """
    Compute distance between transforms (rotation + translation + extra).
    
    Args:
        tf1: First transform
        tf2: Second transform
        reduce: If True, return mean distances as floats
        rotation_weight: Weight for rotation loss in total
        translation_weight: Weight for translation loss in total
        extra_weight: Weight for extra loss in total (e.g., gripper width)
        degrees: If True, return rotation angle in degrees
    
    Returns:
        Tuple of (total_loss, rotation_loss, translation_loss, extra_loss)
        If reduce=True: all as floats
        If reduce=False: all as arrays
        Note: extra_loss is 0.0 if neither transform has extra
    
    Example:
        >>> pred = Transform.from_rep(pred_traj, from_rep="rotation_6d", extra_dims=1)
        >>> target = Transform.from_rep(target_traj, from_rep="rotation_6d", extra_dims=1)
        >>> total, rot_loss, trans_loss, extra_loss = transform_distance(pred, target)
    """
    rot_dist = geodesic_distance(tf1.rotation, tf2.rotation, reduce=False, degrees=degrees)
    trans_dist = translation_distance(tf1.translation, tf2.translation, reduce=False)
    
    backend = tf1.backend
    
    # Compute extra distance if both have extra
    if tf1.extra is not None and tf2.extra is not None:
        if backend == "torch":
            extra_dist = torch.norm(tf1.extra - tf2.extra, dim=-1)
        else:
            extra_dist = np.linalg.norm(tf1.extra - tf2.extra, axis=-1)
    else:
        # No extra, use zero
        if backend == "torch":
            extra_dist = torch.zeros_like(rot_dist)
        else:
            extra_dist = np.zeros_like(rot_dist)
    
    if backend == "torch":
        total = (
            rotation_weight * rot_dist + 
            translation_weight * trans_dist + 
            extra_weight * extra_dist
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
        rotation_weight * rot_dist + 
        translation_weight * trans_dist + 
        extra_weight * extra_dist
    )
    if reduce:
        return (
            float(np.mean(total)), 
            float(np.mean(rot_dist)), 
            float(np.mean(trans_dist)),
            float(np.mean(extra_dist)),
        )
    return total, rot_dist, trans_dist, extra_dist


@overload
def orthogonalize_rotation(matrix: np.ndarray) -> np.ndarray: ...
@overload
def orthogonalize_rotation(matrix: torch.Tensor) -> torch.Tensor: ...


def orthogonalize_rotation(matrix: ArrayLike) -> ArrayLike:
    """
    Project matrix to SO(3) using SVD.
    
    Useful for correcting accumulated numerical errors.
    
    Args:
        matrix: Approximate rotation matrix (..., 3, 3)
    
    Returns:
        Valid rotation matrix (..., 3, 3)
    """
    backend = _get_backend(matrix)
    
    if backend == "torch":
        U, _, Vh = torch.linalg.svd(matrix)
        R = U @ Vh
        # Ensure proper rotation (det = 1)
        det = torch.det(R)
        fix = torch.where(det < 0, -1.0, 1.0)
        # Broadcast fix to match R's shape
        for _ in range(2):
            fix = fix.unsqueeze(-1)
        return R * fix
    
    # NumPy: handle arbitrary batch dimensions
    batch_shape = matrix.shape[:-2]
    matrix_flat = matrix.reshape(-1, 3, 3)
    
    U, _, Vh = np.linalg.svd(matrix_flat)
    R_flat = U @ Vh
    
    # Ensure proper rotation (det = 1)
    det = np.linalg.det(R_flat)
    # Use broadcasting-safe approach
    fix = np.where(det < 0, -1.0, 1.0)
    R_flat = R_flat * fix[:, np.newaxis, np.newaxis]
    
    return R_flat.reshape(*batch_shape, 3, 3)
