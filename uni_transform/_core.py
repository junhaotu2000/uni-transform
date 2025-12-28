"""
Core utilities: types, constants, and backend-agnostic operations.

This module provides the foundational building blocks used throughout uni_transform.
All internal modules depend on this module.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T", np.ndarray, torch.Tensor)
ArrayLike = Union[np.ndarray, torch.Tensor]
Backend = Literal["numpy", "torch"]


# =============================================================================
# Numerical Constants
# =============================================================================

EPS = 1e-8  # General epsilon for division safety
SMALL_ANGLE_THRESHOLD = 1e-6  # Below this, use Taylor approximations


# =============================================================================
# Enums
# =============================================================================


class RotationRepr(str, Enum):
    """Rotation representation types."""

    EULER = "euler"
    QUAT = "quat"
    MATRIX = "matrix"
    ROTATION_6D = "rotation_6d"
    ROT_VEC = "rot_vec"


class TranslationUnit(str, Enum):
    """Translation unit for Transform class."""

    METER = "m"
    MILLIMETER = "mm"

    @property
    def to_meter_scale(self) -> float:
        """Scale factor to convert to meters."""
        if self == TranslationUnit.METER:
            return 1.0
        elif self == TranslationUnit.MILLIMETER:
            return 0.001
        raise ValueError(f"Unknown unit: {self}")

    @property
    def from_meter_scale(self) -> float:
        """Scale factor to convert from meters."""
        return 1.0 / self.to_meter_scale


class VectorInterpMethod(str, Enum):
    """Interpolation methods for vectors/scalars."""

    LINEAR = "linear"
    MINIMUM_JERK = "minimum_jerk"
    CUBIC_SPLINE = "cubic_spline"


class QuatInterpMethod(str, Enum):
    """Interpolation methods for quaternions."""

    SLERP = "slerp"
    NLERP = "nlerp"


class RotationSeqInterpMethod(str, Enum):
    """Interpolation methods for rotation sequences (multi-point)."""

    SLERP = "slerp"
    NLERP = "nlerp"
    SQUAD = "squad"


class UnitMismatchError(ValueError):
    """Raised when attempting to combine transforms with different translation units."""

    pass


# =============================================================================
# Euler Angle Mappings
# =============================================================================

# Most robot states use RPY order [roll, pitch, yaw],
# but seq like "ZYX" expects [yaw, pitch, roll]
EULER_RPY_TO_SEQ_MAPPING = {
    "ZYX": (2, 1, 0),  # rpy -> ypr
    "XYZ": (0, 1, 2),  # rpy -> rpy
    "YZX": (1, 2, 0),
    "ZXY": (2, 0, 1),
    "XZY": (0, 2, 1),
    "YXZ": (1, 0, 2),
}

EULER_SEQ_TO_RPY_MAPPING = {
    "ZYX": (2, 1, 0),  # ypr -> rpy
    "XYZ": (0, 1, 2),  # rpy -> rpy
    "YZX": (2, 0, 1),
    "ZXY": (1, 2, 0),
    "XZY": (0, 2, 1),
    "YXZ": (1, 0, 2),
}

# Expected dimensions for each rotation representation
REP_ROTATION_DIMS = {
    RotationRepr.EULER: 3,
    RotationRepr.QUAT: 4,
    RotationRepr.ROTATION_6D: 6,
    RotationRepr.ROT_VEC: 3,
}

# Unit conversion factors
UNIT_CONVERSION = {
    (TranslationUnit.METER, TranslationUnit.MILLIMETER): 1000.0,
    (TranslationUnit.MILLIMETER, TranslationUnit.METER): 0.001,
    (TranslationUnit.METER, TranslationUnit.METER): 1.0,
    (TranslationUnit.MILLIMETER, TranslationUnit.MILLIMETER): 1.0,
}


# =============================================================================
# Backend Detection
# =============================================================================


def get_backend(x: ArrayLike) -> Backend:
    """Determine backend from input type."""
    return "torch" if isinstance(x, torch.Tensor) else "numpy"


def to_backend(
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
# Backend-Agnostic Operations
# =============================================================================


def normalize(x: ArrayLike, dim: int = -1, eps: float = EPS) -> ArrayLike:
    """Normalize vectors along specified dimension."""
    if isinstance(x, torch.Tensor):
        return F.normalize(x, dim=dim, eps=eps)
    norm = np.linalg.norm(x, axis=dim, keepdims=True)
    return x / np.maximum(norm, eps)


def cross(a: ArrayLike, b: ArrayLike, dim: int = -1) -> ArrayLike:
    """Cross product along specified dimension."""
    if isinstance(a, torch.Tensor):
        return torch.cross(a, b, dim=dim)
    return np.cross(a, b, axis=dim)


def dot_keepdim(a: ArrayLike, b: ArrayLike, dim: int = -1) -> ArrayLike:
    """Dot product keeping dimension."""
    if isinstance(a, torch.Tensor):
        return (a * b).sum(dim, keepdim=True)
    return np.sum(a * b, axis=dim, keepdims=True)


def cat(arrays: List[ArrayLike], dim: int = -1) -> ArrayLike:
    """Concatenate arrays along dimension."""
    if isinstance(arrays[0], torch.Tensor):
        return torch.cat(arrays, dim=dim)
    return np.concatenate(arrays, axis=dim)


def stack(arrays: List[ArrayLike], dim: int = -1) -> ArrayLike:
    """Stack arrays along new dimension."""
    if isinstance(arrays[0], torch.Tensor):
        return torch.stack(arrays, dim=dim)
    return np.stack(arrays, axis=dim)


def take_indices(x: ArrayLike, indices: Tuple[int, ...], dim: int = -1) -> ArrayLike:
    """Index selection with cached tensors for PyTorch."""
    if isinstance(x, torch.Tensor):
        idx_tensor = _cached_indices(indices, str(x.device))
        return x.index_select(dim, idx_tensor)
    if dim == -1:
        return x[..., list(indices)]
    return np.take(x, list(indices), axis=dim)


def transpose_last_two(x: ArrayLike) -> ArrayLike:
    """Transpose last two dimensions."""
    if isinstance(x, torch.Tensor):
        return x.transpose(-1, -2)
    return np.swapaxes(x, -1, -2)


def matmul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Matrix multiplication."""
    if isinstance(a, torch.Tensor):
        return torch.matmul(a, b)
    return np.matmul(a, b)


def eye(n: int, backend: Backend, dtype=None, device=None) -> ArrayLike:
    """Identity matrix."""
    if backend == "torch":
        return torch.eye(n, dtype=dtype, device=device)
    return np.eye(n, dtype=dtype or np.float64)


def zeros(shape: Tuple[int, ...], backend: Backend, dtype=None, device=None) -> ArrayLike:
    """Zero tensor/array."""
    if backend == "torch":
        return torch.zeros(shape, dtype=dtype, device=device)
    return np.zeros(shape, dtype=dtype or np.float64)

