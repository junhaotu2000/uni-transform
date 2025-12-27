"""
Rotation class with unified from_rep/to_rep API.

Provides an object-oriented interface for rotation operations,
supporting both NumPy and PyTorch backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import torch

from ._core import (
    ArrayLike,
    Backend,
    RotationRepr,
    eye,
    get_backend,
    matmul,
    transpose_last_two,
)
from .rotation_conversions import (
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_quaternion,
    matrix_to_rotation,
    matrix_to_rotation_6d,
    matrix_to_rotvec,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    rotation_to_matrix,
    rotvec_to_matrix,
)


@dataclass(slots=True, eq=False)
class Rotation:
    """
    Rotation class with unified from_rep/to_rep API.

    Supports both NumPy and PyTorch backends with arbitrary batch dimensions.
    Internal representation: rotation matrix (..., 3, 3)

    Example:
        >>> # Create from quaternion
        >>> rot = Rotation.from_rep(quat, from_rep="quat")
        >>> euler = rot.to_rep("euler", seq="ZYX")

        >>> # Compose rotations
        >>> rot_combined = rot1 @ rot2

        >>> # Apply to vectors
        >>> rotated_v = rot.apply(v)

        >>> # Batch operations
        >>> batch_rot = Rotation.from_rep(batch_quats, from_rep="quat")
    """

    matrix: ArrayLike  # (..., 3, 3)
    backend: Backend = field(init=False)

    def __post_init__(self) -> None:
        """Validate and detect backend."""
        if self.matrix.shape[-2:] != (3, 3):
            raise ValueError(
                f"Rotation matrix must have shape (..., 3, 3), got {self.matrix.shape}"
            )
        self.backend = get_backend(self.matrix)

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def identity(
        cls,
        backend: Backend = "numpy",
        dtype=None,
        device=None,
    ) -> "Rotation":
        """Create identity rotation."""
        return cls(matrix=eye(3, backend, dtype=dtype, device=device))

    @classmethod
    def from_rep(
        cls,
        rotation: ArrayLike,
        *,
        from_rep: Union[str, RotationRepr],
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
    ) -> "Rotation":
        """
        Create rotation from any representation.

        Args:
            rotation: Rotation in source representation
            from_rep: Source representation ("euler", "quat", "matrix", "rotation_6d", "rot_vec")
            seq: Euler sequence (only used if from_rep is "euler")
            degrees: If True, euler angles are in degrees
            euler_in_rpy: If True and from_rep is euler, input is in [r,p,y] order

        Returns:
            Rotation instance
        """
        matrix = rotation_to_matrix(
            rotation, from_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )
        return cls(matrix=matrix)

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> "Rotation":
        """Create from rotation matrix (..., 3, 3)."""
        return cls(matrix=matrix)

    @classmethod
    def from_quat(cls, quat: ArrayLike) -> "Rotation":
        """Create from quaternion (xyzw format)."""
        return cls(matrix=quaternion_to_matrix(quat))

    @classmethod
    def from_euler(
        cls,
        euler: ArrayLike,
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
    ) -> "Rotation":
        """Create from euler angles."""
        return cls(
            matrix=euler_to_matrix(euler, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)
        )

    @classmethod
    def from_rotvec(cls, rotvec: ArrayLike) -> "Rotation":
        """Create from rotation vector (axis-angle)."""
        return cls(matrix=rotvec_to_matrix(rotvec))

    @classmethod
    def from_rotation_6d(cls, rot_6d: ArrayLike) -> "Rotation":
        """Create from 6D rotation representation."""
        return cls(matrix=rotation_6d_to_matrix(rot_6d))

    @classmethod
    def stack(cls, rotations: List["Rotation"], axis: int = 0) -> "Rotation":
        """
        Stack multiple rotations into a batched rotation.

        Args:
            rotations: List of Rotation objects
            axis: Axis along which to stack (default: 0)

        Returns:
            Batched Rotation
        """
        if not rotations:
            raise ValueError("Cannot stack empty list of rotations")

        backend = rotations[0].backend
        if backend == "numpy":
            matrix = np.stack([r.matrix for r in rotations], axis=axis)
        else:
            matrix = torch.stack([r.matrix for r in rotations], dim=axis)

        return cls(matrix=matrix)

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_rep(
        self,
        to_rep: Union[str, RotationRepr],
        *,
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
    ) -> ArrayLike:
        """
        Convert to any rotation representation.

        Args:
            to_rep: Target representation ("euler", "quat", "matrix", "rotation_6d", "rot_vec")
            seq: Euler sequence (only used if to_rep is "euler")
            degrees: If True, return euler angles in degrees
            euler_in_rpy: If True and to_rep is euler, output is in [r,p,y] order

        Returns:
            Rotation in target representation
        """
        return matrix_to_rotation(
            self.matrix, to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )

    def as_matrix(self) -> ArrayLike:
        """Return rotation matrix (..., 3, 3)."""
        return self.matrix

    def as_quat(self) -> ArrayLike:
        """Return quaternion (xyzw format)."""
        return matrix_to_quaternion(self.matrix)

    def as_euler(
        self,
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
    ) -> ArrayLike:
        """Return euler angles."""
        return matrix_to_euler(self.matrix, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)

    def as_rotvec(self) -> ArrayLike:
        """Return rotation vector (axis-angle)."""
        return matrix_to_rotvec(self.matrix)

    def as_rotation_6d(self) -> ArrayLike:
        """Return 6D rotation representation."""
        return matrix_to_rotation_6d(self.matrix)

    # -------------------------------------------------------------------------
    # Rotation Operations
    # -------------------------------------------------------------------------

    def __matmul__(self, other: "Rotation") -> "Rotation":
        """
        Compose rotations: self @ other.

        The result rotation applies other first, then self.
        """
        if self.backend != other.backend:
            raise ValueError(
                f"Cannot compose rotations with different backends: "
                f"{self.backend} vs {other.backend}"
            )
        return Rotation(matrix=matmul(self.matrix, other.matrix))

    def compose(self, other: "Rotation") -> "Rotation":
        """Alias for @ operator."""
        return self @ other

    def inverse(self) -> "Rotation":
        """Compute inverse rotation (transpose of matrix)."""
        return Rotation(matrix=transpose_last_two(self.matrix))

    def apply(self, vectors: ArrayLike) -> ArrayLike:
        """
        Apply rotation to vector(s).

        Args:
            vectors: Vector(s) to rotate (..., 3)

        Returns:
            Rotated vector(s) (..., 3)
        """
        return matmul(self.matrix, vectors[..., None]).squeeze(-1)

    def apply_delta(self, delta: "Rotation", in_body_frame: bool = False) -> "Rotation":
        """
        Apply a delta (incremental) rotation.

        Args:
            delta: Incremental rotation to apply
            in_body_frame: If True, delta is in body frame (result = self @ delta)
                          If False, delta is in world frame (result = delta @ self)

        Returns:
            Updated Rotation
        """
        if in_body_frame:
            return self @ delta
        else:
            return delta @ self

    def relative_to(self, reference: "Rotation") -> "Rotation":
        """
        Compute this rotation relative to a reference frame.

        Returns R such that: reference @ R = self
        i.e., R = reference.inverse() @ self

        Args:
            reference: Reference rotation frame

        Returns:
            Rotation relative to reference
        """
        return reference.inverse() @ self

    # -------------------------------------------------------------------------
    # Interpolation
    # -------------------------------------------------------------------------

    def slerp(self, other: "Rotation", t: Union[float, ArrayLike]) -> "Rotation":
        """
        Spherical linear interpolation to another rotation.

        Args:
            other: Target rotation
            t: Interpolation parameter(s) in [0, 1]. t=0 returns self, t=1 returns other.

        Returns:
            Interpolated Rotation
        """
        from .interpolation import quaternion_slerp

        q0 = self.as_quat()
        q1 = other.as_quat()
        q_interp = quaternion_slerp(q0, q1, t)
        return Rotation.from_quat(q_interp)

    def nlerp(self, other: "Rotation", t: Union[float, ArrayLike]) -> "Rotation":
        """
        Normalized linear interpolation to another rotation (faster than slerp).

        Args:
            other: Target rotation
            t: Interpolation parameter(s) in [0, 1]

        Returns:
            Interpolated Rotation
        """
        from .interpolation import quaternion_nlerp

        q0 = self.as_quat()
        q1 = other.as_quat()
        q_interp = quaternion_nlerp(q0, q1, t)
        return Rotation.from_quat(q_interp)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Get batch dimensions."""
        return self.matrix.shape[:-2]

    @property
    def is_batched(self) -> bool:
        """Check if rotation has batch dimensions."""
        return len(self.batch_shape) > 0

    @property
    def device(self):
        """Get device (for PyTorch tensors)."""
        if self.backend == "torch":
            return self.matrix.device
        return None

    @property
    def dtype(self):
        """Get data type."""
        return self.matrix.dtype

    @property
    def requires_grad(self) -> bool:
        """Check if gradients are enabled (PyTorch only)."""
        if self.backend == "torch":
            return self.matrix.requires_grad
        return False

    def requires_grad_(self, requires_grad: bool = True) -> "Rotation":
        """Enable/disable gradient tracking in-place (PyTorch only)."""
        if self.backend != "torch":
            raise ValueError("requires_grad_() is only supported for PyTorch tensors")
        self.matrix.requires_grad_(requires_grad)
        return self

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def to(self, device=None, dtype=None) -> "Rotation":
        """Move rotation to device/dtype (PyTorch only)."""
        if self.backend != "torch":
            raise ValueError("to() is only supported for PyTorch tensors")
        return Rotation(matrix=self.matrix.to(device=device, dtype=dtype))

    def detach(self) -> "Rotation":
        """Detach from computation graph (PyTorch only)."""
        if self.backend != "torch":
            return self
        return Rotation(matrix=self.matrix.detach())

    def clone(self) -> "Rotation":
        """Create a copy of the rotation."""
        if self.backend == "torch":
            return Rotation(matrix=self.matrix.clone())
        return Rotation(matrix=self.matrix.copy())

    def __repr__(self) -> str:
        """String representation."""
        return f"Rotation(backend={self.backend}, batch_shape={self.batch_shape}, dtype={self.dtype})"

    def __getitem__(self, idx) -> "Rotation":
        """Index into batched rotations."""
        return Rotation(matrix=self.matrix[idx])

    # -------------------------------------------------------------------------
    # Static Conversion Helper
    # -------------------------------------------------------------------------

    @staticmethod
    def convert(
        rotation: ArrayLike,
        *,
        from_rep: Union[str, RotationRepr],
        to_rep: Union[str, RotationRepr],
        seq: str = "ZYX",
        degrees: bool = False,
        euler_in_rpy: bool = False,
    ) -> ArrayLike:
        """
        Convert rotation between representations without creating Rotation instance.

        Args:
            rotation: Rotation in source representation
            from_rep: Source representation
            to_rep: Target representation
            seq: Euler sequence (for euler representations)
            degrees: If True, euler angles are in degrees
            euler_in_rpy: If True and euler, use [r,p,y] order

        Returns:
            Rotation in target representation
        """
        from .rotation_conversions import convert_rotation

        return convert_rotation(
            rotation,
            from_rep=from_rep,
            to_rep=to_rep,
            seq=seq,
            degrees=degrees,
            euler_in_rpy=euler_in_rpy,
        )

