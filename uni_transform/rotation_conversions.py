"""
Rotation conversion functions between different representations.

Supported representations:
- quaternion (xyzw convention, matching SciPy/ROS)
- euler angles (configurable sequence, default ZYX)
- rotation matrix (3x3)
- rotation vector / axis-angle
- 6D rotation (continuous representation for learning)

All functions support:
- NumPy arrays and PyTorch tensors
- Arbitrary batch dimensions
- Differentiable operations (PyTorch)
"""

from __future__ import annotations

from typing import Callable, Dict, Union, overload

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from ._core import (
    ArrayLike,
    EULER_RPY_TO_SEQ_MAPPING,
    EULER_SEQ_TO_RPY_MAPPING,
    RotationRepr,
    SMALL_ANGLE_THRESHOLD,
    cat,
    cross,
    get_backend,
    normalize,
    stack,
    take_indices,
)


# =============================================================================
# 6D Rotation Representation
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
    return cat([row0, row1], dim=-1)


@overload
def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray: ...
@overload
def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor: ...


def rotation_6d_to_matrix(rot_6d: ArrayLike) -> ArrayLike:
    """
    Convert 6D rotation to rotation matrix using Gram-Schmidt orthogonalization.

    Args:
        rot_6d: 6D rotation representation (..., 6) as [row0, row1]

    Returns:
        Rotation matrix (..., 3, 3)

    Raises:
        ValueError: If input shape is invalid
    """
    if rot_6d.shape[-1] != 6:
        raise ValueError(f"6D rotation must have shape (..., 6), got {rot_6d.shape}")

    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = normalize(a1, dim=-1)
    b3 = cross(b1, a2, dim=-1)
    b3 = normalize(b3, dim=-1)
    b2 = cross(b3, b1, dim=-1)

    return stack([b1, b2, b3], dim=-2)


# =============================================================================
# Quaternion Operations (xyzw convention)
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
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have shape (..., 4), got {quat.shape}")

    backend = get_backend(quat)

    if backend == "numpy":
        batch_shape = quat.shape[:-1]
        quat_flat = quat.reshape(-1, 4)
        matrix_flat = ScipyRotation.from_quat(quat_flat).as_matrix()
        return matrix_flat.reshape(*batch_shape, 3, 3)

    # PyTorch implementation
    x, y, z, w = torch.unbind(quat, dim=-1)

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
    backend = get_backend(matrix)

    if backend == "numpy":
        batch_shape = matrix.shape[:-2]
        matrix_flat = matrix.reshape(-1, 3, 3)
        quat_flat = ScipyRotation.from_matrix(matrix_flat).as_quat()
        return quat_flat.reshape(*batch_shape, 4)

    # PyTorch: Shepperd's method
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    quat = torch.zeros(m.shape[0], 4, dtype=matrix.dtype, device=matrix.device)

    # Shepperd's method with proper EPS inside sqrt to prevent divide-by-zero
    # Case 1: trace > 0 (trace + 1 >= 1, always safe)
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        quat[mask1, 3] = 0.25 * s
        quat[mask1, 0] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
        quat[mask1, 1] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s
        quat[mask1, 2] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s

    # Case 2: m[0,0] is largest (clamp inside sqrt)
    mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if mask2.any():
        val = torch.clamp(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2], min=SMALL_ANGLE_THRESHOLD)
        s = torch.sqrt(val) * 2
        quat[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s
        quat[mask2, 0] = 0.25 * s
        quat[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
        quat[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s

    # Case 3: m[1,1] is largest
    mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    if mask3.any():
        val = torch.clamp(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2], min=SMALL_ANGLE_THRESHOLD)
        s = torch.sqrt(val) * 2
        quat[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s
        quat[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s
        quat[mask3, 1] = 0.25 * s
        quat[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s

    # Case 4: m[2,2] is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        val = torch.clamp(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1], min=SMALL_ANGLE_THRESHOLD)
        s = torch.sqrt(val) * 2
        quat[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s
        quat[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s
        quat[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s
        quat[mask4, 2] = 0.25 * s

    # Ensure positive w (canonical form)
    quat = torch.where(quat[:, 3:4] < 0, -quat, quat)

    return quat.reshape(batch_shape + (4,))


# Pre-allocated for common NumPy dtypes (no lookup overhead)
_CONJ_SIGN_F32 = np.array([-1, -1, -1, 1], dtype=np.float32)
_CONJ_SIGN_F64 = np.array([-1, -1, -1, 1], dtype=np.float64)


def quaternion_conjugate(q: ArrayLike) -> ArrayLike:
    """Compute quaternion conjugate. For unit quaternions, equals inverse."""
    if isinstance(q, torch.Tensor):
        # Creating a 4-element tensor is cheap. Don't overthink it.
        return q * torch.tensor([-1, -1, -1, 1], dtype=q.dtype, device=q.device)
    # NumPy: use pre-allocated for common dtypes
    if q.dtype == np.float32:
        return q * _CONJ_SIGN_F32
    if q.dtype == np.float64:
        return q * _CONJ_SIGN_F64
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


@overload
def quaternion_inverse(q: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_inverse(q: torch.Tensor) -> torch.Tensor: ...


def quaternion_inverse(q: ArrayLike) -> ArrayLike:
    """
    Compute quaternion inverse.

    For unit quaternions, this is equivalent to conjugate.

    Args:
        q: Quaternion(s) in xyzw format (..., 4)

    Returns:
        Inverse quaternion(s) (..., 4)
    """
    conj = quaternion_conjugate(q)

    if isinstance(q, torch.Tensor):
        norm_sq = (q * q).sum(dim=-1, keepdim=True)
        return conj / (norm_sq + SMALL_ANGLE_THRESHOLD)

    norm_sq = np.sum(q * q, axis=-1, keepdims=True)
    return conj / (norm_sq + SMALL_ANGLE_THRESHOLD)


@overload
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray: ...
@overload
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor: ...


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> ArrayLike:
    """
    Multiply two quaternions (Hamilton product).

    The result represents the composition of rotations: first q2, then q1.

    Args:
        q1: First quaternion(s) in xyzw format (..., 4)
        q2: Second quaternion(s) in xyzw format (..., 4)

    Returns:
        Product quaternion(s) in xyzw format (..., 4)
    """
    backend = get_backend(q1)

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

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

    Args:
        q: Quaternion(s) in xyzw format (..., 4)
        v: Vector(s) to rotate (..., 3)

    Returns:
        Rotated vector(s) (..., 3)
    """
    backend = get_backend(q)

    qxyz = q[..., :3]
    qw = q[..., 3:4]

    if backend == "numpy":
        t = 2 * np.cross(qxyz, v, axis=-1)
        return v + qw * t + np.cross(qxyz, t, axis=-1)

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
    euler: np.ndarray,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> np.ndarray: ...
@overload
def euler_to_matrix(
    euler: torch.Tensor,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
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
        euler_in_rpy: If True, input is in [roll, pitch, yaw] order

    Returns:
        Rotation matrix (..., 3, 3)
    """
    backend = get_backend(euler)

    if euler_in_rpy and seq in EULER_RPY_TO_SEQ_MAPPING:
        idx = EULER_RPY_TO_SEQ_MAPPING[seq]
        euler = take_indices(euler, idx, dim=-1)

    if backend == "numpy":
        batch_shape = euler.shape[:-1]
        euler_flat = euler.reshape(-1, 3)
        matrix_flat = ScipyRotation.from_euler(seq, euler_flat, degrees=degrees).as_matrix()
        return matrix_flat.reshape(*batch_shape, 3, 3)

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
    matrix: np.ndarray,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
) -> np.ndarray: ...
@overload
def matrix_to_euler(
    matrix: torch.Tensor,
    seq: str = "ZYX",
    degrees: bool = False,
    euler_in_rpy: bool = False,
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

    Args:
        matrix: Rotation matrix (..., 3, 3)
        seq: Rotation sequence (e.g., "ZYX", "XYZ")
        degrees: If True, return angles in degrees
        euler_in_rpy: If True, output is in [roll, pitch, yaw] order

    Returns:
        Euler angles (..., 3)
    """
    backend = get_backend(matrix)

    if backend == "numpy":
        batch_shape = matrix.shape[:-2]
        matrix_flat = matrix.reshape(-1, 3, 3)
        euler_flat = ScipyRotation.from_matrix(matrix_flat).as_euler(seq, degrees=degrees)
        euler = euler_flat.reshape(*batch_shape, 3)

        if euler_in_rpy and seq in EULER_SEQ_TO_RPY_MAPPING:
            idx = list(EULER_SEQ_TO_RPY_MAPPING[seq])
            euler = euler[..., idx]
        return euler

    # PyTorch: use scipy via numpy (euler extraction has gimbal lock issues)
    quat = matrix_to_quaternion(matrix)
    quat_np = quat.detach().cpu().numpy()
    batch_shape = quat_np.shape[:-1]
    quat_flat = quat_np.reshape(-1, 4)
    euler_flat = ScipyRotation.from_quat(quat_flat).as_euler(seq, degrees=degrees)
    euler_np = euler_flat.reshape(*batch_shape, 3)

    if euler_in_rpy and seq in EULER_SEQ_TO_RPY_MAPPING:
        idx = list(EULER_SEQ_TO_RPY_MAPPING[seq])
        euler_np = euler_np[..., idx]

    return torch.as_tensor(euler_np, dtype=matrix.dtype, device=matrix.device)


def _matrix_to_euler_torch_differentiable(
    matrix: torch.Tensor,
    seq: str = "ZYX",
    degrees: bool = False,
) -> torch.Tensor:
    """
    Pure PyTorch differentiable euler angle extraction.

    Warning: May have numerical issues near gimbal lock.
    """
    axis_to_idx = {"X": 0, "Y": 1, "Z": 2}

    i = axis_to_idx[seq[0]]
    j = axis_to_idx[seq[1]]
    k = axis_to_idx[seq[2]]

    is_tait_bryan = i != k

    if not is_tait_bryan:
        raise NotImplementedError(
            f"Proper Euler sequence {seq} not implemented. "
            "Use Tait-Bryan sequences (ZYX, XYZ, etc.)."
        )

    sign = 1.0 if (j - i) % 3 == 1 else -1.0
    sin_angle2 = sign * matrix[..., i, k]
    sin_angle2 = torch.clamp(sin_angle2, -1.0, 1.0)
    angle2 = torch.asin(sin_angle2)

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

    if degrees:
        euler = torch.rad2deg(euler)

    return euler


@overload
def matrix_to_euler_differentiable(
    matrix: np.ndarray,
    seq: str = "ZYX",
    degrees: bool = False,
) -> np.ndarray: ...
@overload
def matrix_to_euler_differentiable(
    matrix: torch.Tensor,
    seq: str = "ZYX",
    degrees: bool = False,
) -> torch.Tensor: ...


def matrix_to_euler_differentiable(
    matrix: ArrayLike,
    seq: str = "ZYX",
    degrees: bool = False,
) -> ArrayLike:
    """
    Differentiable euler angle extraction with gradient support.

    Warning: May have numerical issues near gimbal lock singularities.

    Args:
        matrix: Rotation matrix (..., 3, 3)
        seq: Euler sequence (ZYX, XYZ, YZX, ZXY, XZY, YXZ)
        degrees: If True, return angles in degrees

    Returns:
        Euler angles (..., 3)
    """
    if isinstance(matrix, torch.Tensor):
        return _matrix_to_euler_torch_differentiable(matrix, seq, degrees)

    batch_shape = matrix.shape[:-2]
    matrix_flat = matrix.reshape(-1, 3, 3)
    euler_flat = ScipyRotation.from_matrix(matrix_flat).as_euler(seq, degrees=degrees)
    return euler_flat.reshape(*batch_shape, 3)


# =============================================================================
# Rotation Vector (Axis-Angle)
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
    backend = get_backend(rotvec)

    if backend == "numpy":
        batch_shape = rotvec.shape[:-1]
        rotvec_flat = rotvec.reshape(-1, 3)
        matrix_flat = ScipyRotation.from_rotvec(rotvec_flat).as_matrix()
        return matrix_flat.reshape(*batch_shape, 3, 3)

    # PyTorch: Rodrigues formula
    batch_shape = rotvec.shape[:-1]
    rotvec_flat = rotvec.reshape(-1, 3)

    angle = torch.norm(rotvec_flat, dim=-1, keepdim=True)
    axis = rotvec_flat / torch.clamp(angle, min=1e-8)

    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)

    x, y, z = torch.unbind(axis, dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)

    I = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    R = I + sin_a * K + (1 - cos_a) * torch.matmul(K, K)

    small_angle = angle.squeeze(-1) < SMALL_ANGLE_THRESHOLD
    if small_angle.any():
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
    backend = get_backend(matrix)

    if backend == "numpy":
        batch_shape = matrix.shape[:-2]
        matrix_flat = matrix.reshape(-1, 3, 3)
        rotvec_flat = ScipyRotation.from_matrix(matrix_flat).as_rotvec()
        return rotvec_flat.reshape(*batch_shape, 3)

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
    """
    backend = get_backend(quat)

    if backend == "numpy":
        batch_shape = quat.shape[:-1]
        quat_flat = quat.reshape(-1, 4)
        rotvec_flat = ScipyRotation.from_quat(quat_flat).as_rotvec()
        return rotvec_flat.reshape(*batch_shape, 3)

    sign = torch.sign(quat[..., 3:4])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    xyz = quat[..., :3] * sign
    w = quat[..., 3:4] * sign

    norm = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, w)

    scale = torch.where(
        norm > SMALL_ANGLE_THRESHOLD,
        angle / norm,
        2 * torch.ones_like(angle),
    )

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
    backend = get_backend(rotvec)

    if backend == "numpy":
        batch_shape = rotvec.shape[:-1]
        rotvec_flat = rotvec.reshape(-1, 3)
        quat_flat = ScipyRotation.from_rotvec(rotvec_flat).as_quat()
        return quat_flat.reshape(*batch_shape, 4)

    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    half_angle = angle / 2

    scale = torch.where(
        angle > SMALL_ANGLE_THRESHOLD,
        torch.sin(half_angle) / angle,
        0.5 * torch.ones_like(angle),
    )

    xyz = rotvec * scale
    w = torch.cos(half_angle)

    return torch.cat([xyz, w], dim=-1)


# =============================================================================
# Generic Conversion Dispatch
# =============================================================================

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
    matrix = rotation_to_matrix(
        rotation, from_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
    )
    return matrix_to_rotation(
        matrix, to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
    )


# =============================================================================
# Utility Functions
# =============================================================================


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
    backend = get_backend(matrix)

    if backend == "torch":
        U, _, Vh = torch.linalg.svd(matrix)
        R = U @ Vh
        det = torch.det(R)
        fix = torch.where(det < 0, -1.0, 1.0)
        for _ in range(2):
            fix = fix.unsqueeze(-1)
        return R * fix

    batch_shape = matrix.shape[:-2]
    matrix_flat = matrix.reshape(-1, 3, 3)

    U, _, Vh = np.linalg.svd(matrix_flat)
    R_flat = U @ Vh

    det = np.linalg.det(R_flat)
    fix = np.where(det < 0, -1.0, 1.0)
    R_flat = R_flat * fix[:, np.newaxis, np.newaxis]

    return R_flat.reshape(*batch_shape, 3, 3)


def xyz_rotation_6d_to_matrix(xyz_rot_6d: ArrayLike) -> ArrayLike:
    """
    Convert [x, y, z, 6D rotation] to 4x4 homogeneous transformation matrix.

    Args:
        xyz_rot_6d: Array of shape (..., 9) containing [x, y, z, rot6d]

    Returns:
        Homogeneous matrix (..., 4, 4)
    """
    # Import here to avoid circular dependency
    from .transform import Transform

    if xyz_rot_6d.shape[-1] != 9:
        raise ValueError(f"Expected last dimension 9, got {xyz_rot_6d.shape[-1]}")

    translation = xyz_rot_6d[..., :3]
    rot_6d = xyz_rot_6d[..., 3:9]
    rotation = rotation_6d_to_matrix(rot_6d)

    return Transform(rotation=rotation, translation=translation).as_matrix()

