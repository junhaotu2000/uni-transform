"""
Transform class for rigid body transformations (rotation + translation).

Provides an object-oriented interface for SE(3) transformations,
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
    REP_ROTATION_DIMS,
    RotationRepr,
    TranslationUnit,
    UNIT_CONVERSION,
    UnitMismatchError,
    cat,
    eye,
    get_backend,
    matmul,
    transpose_last_two,
    zeros,
)
from .rotation_conversions import (
    matrix_to_rotation,
    quaternion_to_matrix,
    rotation_to_matrix,
)


@dataclass(slots=True, eq=False)
class Transform:
    """
    Rigid body transform supporting both NumPy and PyTorch backends.

    Attributes:
        rotation: Rotation matrix (..., 3, 3)
        translation: Translation vector (..., 3)
        translation_unit: Unit of translation ("m" for meters, "mm" for millimeters)
        extra: Optional extra state like gripper width (..., K)
        backend: "numpy" or "torch" (auto-detected from inputs)

    Example:
        >>> tf = Transform.from_rep(
        ...     np.array([1, 2, 3, 0, 0, 0, 1, 0.04]),
        ...     from_rep="quat",
        ...     extra_dims=1,
        ...     translation_unit="m",
        ... )
        >>> tf_mm = tf.to_unit("mm")
        >>> pose = tf.to_rep("euler")
    """

    rotation: ArrayLike
    translation: ArrayLike
    translation_unit: TranslationUnit = TranslationUnit.METER
    extra: ArrayLike = None
    backend: Backend = field(init=False)

    def __post_init__(self) -> None:
        """Validate and normalize inputs."""
        if isinstance(self.translation_unit, str):
            self.translation_unit = TranslationUnit(self.translation_unit)

        rot_backend = get_backend(self.rotation)
        trans_backend = get_backend(self.translation)
        extra_backend = get_backend(self.extra) if self.extra is not None else None

        is_torch = (
            rot_backend == "torch"
            or trans_backend == "torch"
            or extra_backend == "torch"
        )
        self.backend = "torch" if is_torch else "numpy"

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
        translation_unit: Union[str, TranslationUnit] = TranslationUnit.METER,
    ) -> "Transform":
        """Create identity transform with optional extra dimensions."""
        rot = eye(3, backend, dtype=dtype, device=device)
        trans = zeros((3,), backend, dtype=dtype, device=device)
        extra = zeros((extra_dims,), backend, dtype=dtype, device=device) if extra_dims > 0 else None
        return cls(rotation=rot, translation=trans, translation_unit=translation_unit, extra=extra)

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        translation_unit: Union[str, TranslationUnit] = TranslationUnit.METER,
    ) -> "Transform":
        """Create transform from 4x4 homogeneous matrix."""
        return cls(
            rotation=matrix[..., :3, :3],
            translation=matrix[..., :3, 3],
            translation_unit=translation_unit,
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
        translation_unit: Union[str, TranslationUnit] = TranslationUnit.METER,
    ) -> "Transform":
        """
        Create transform from translation + rotation representation + optional extra.

        Args:
            tf: [translation (3), rotation (varies by repr), extra (extra_dims)]
            from_rep: Rotation representation
            seq: Euler sequence (if euler)
            degrees: If euler, whether angles are in degrees
            euler_in_rpy: If True and from_rep is euler, input is in [r,p,y] order
            extra_dims: Number of extra dimensions at the end
            requires_grad: If True, enable gradient tracking (PyTorch only)
            translation_unit: Unit of translation ("m" or "mm")

        Returns:
            Transform instance
        """
        from_rep = RotationRepr(from_rep)

        if extra_dims < 0:
            raise ValueError(f"extra_dims must be non-negative, got {extra_dims}")

        if from_rep != RotationRepr.MATRIX:
            rot_dims = REP_ROTATION_DIMS.get(from_rep, 0)
            min_dims = 3 + rot_dims
            total_dims = tf.shape[-1]
            expected_extra = total_dims - min_dims

            if extra_dims > expected_extra:
                raise ValueError(
                    f"extra_dims={extra_dims} exceeds available dimensions. "
                    f"Input has {total_dims} dims, {from_rep.value} requires "
                    f"{min_dims} (3 translation + {rot_dims} rotation), "
                    f"leaving {expected_extra} for extra."
                )

        extra = None
        if extra_dims > 0:
            extra = tf[..., -extra_dims:]
            tf = tf[..., :-extra_dims]

        if from_rep == RotationRepr.MATRIX:
            result = cls.from_matrix(tf, translation_unit=translation_unit)
            result.extra = extra
            if requires_grad and result.backend == "torch":
                result.requires_grad_(True)
            return result

        translation = tf[..., :3]
        rotation_part = tf[..., 3:]
        rotation = rotation_to_matrix(
            rotation_part, from_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )

        result = cls(
            rotation=rotation,
            translation=translation,
            translation_unit=translation_unit,
            extra=extra,
        )
        if requires_grad and result.backend == "torch":
            result.requires_grad_(True)
        return result

    @classmethod
    def from_pos_quat(
        cls,
        position: ArrayLike,
        quaternion: ArrayLike,
        extra: ArrayLike = None,
        translation_unit: Union[str, TranslationUnit] = TranslationUnit.METER,
    ) -> "Transform":
        """
        Create transform from position and quaternion.

        Args:
            position: Translation vector (..., 3)
            quaternion: Quaternion in xyzw format (..., 4)
            extra: Optional extra state (..., K)
            translation_unit: Unit of translation
        """
        rotation = quaternion_to_matrix(quaternion)
        return cls(
            rotation=rotation,
            translation=position,
            translation_unit=translation_unit,
            extra=extra,
        )

    @classmethod
    def stack(cls, transforms: List["Transform"], axis: int = 0) -> "Transform":
        """
        Stack multiple transforms into a batched transform.

        All transforms must have the same translation unit.
        """
        if not transforms:
            raise ValueError("Cannot stack empty list of transforms")

        first_unit = transforms[0].translation_unit
        for i, tf in enumerate(transforms[1:], 1):
            if tf.translation_unit != first_unit:
                raise UnitMismatchError(
                    f"Cannot stack transforms with different translation units: "
                    f"transforms[0] has {first_unit.value}, "
                    f"transforms[{i}] has {tf.translation_unit.value}."
                )

        backend = transforms[0].backend
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

        return cls(
            rotation=rotation,
            translation=translation,
            translation_unit=first_unit,
            extra=extra,
        )

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
        """
        to_rep = RotationRepr(to_rep)

        if to_rep == RotationRepr.MATRIX:
            return self.as_matrix()

        rotation_repr = matrix_to_rotation(
            self.rotation, to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy
        )
        result = cat([self.translation, rotation_repr], dim=-1)

        if include_extra and self.extra is not None:
            result = cat([result, self.extra], dim=-1)

        return result

    # -------------------------------------------------------------------------
    # Transform Operations
    # -------------------------------------------------------------------------

    def __matmul__(self, other: "Transform") -> "Transform":
        """
        Compose transforms: self @ other.

        The result transforms points as: self.transform(other.transform(point))
        """
        if self.backend != other.backend:
            raise ValueError(
                f"Cannot compose transforms with different backends: "
                f"{self.backend} vs {other.backend}"
            )

        if self.translation_unit != other.translation_unit:
            raise UnitMismatchError(
                f"Cannot compose transforms with different translation units: "
                f"{self.translation_unit.value} vs {other.translation_unit.value}."
            )

        rotation = matmul(self.rotation, other.rotation)
        translation = (
            matmul(self.rotation, other.translation[..., None]).squeeze(-1)
            + self.translation
        )
        extra = other.extra if other.extra is not None else self.extra

        return Transform(
            rotation=rotation,
            translation=translation,
            translation_unit=self.translation_unit,
            extra=extra,
        )

    def compose(self, other: "Transform") -> "Transform":
        """Alias for @ operator."""
        return self @ other

    def inverse(self) -> "Transform":
        """Compute inverse transform."""
        rot_inv = transpose_last_two(self.rotation)
        trans_inv = -matmul(rot_inv, self.translation[..., None]).squeeze(-1)
        return Transform(
            rotation=rot_inv,
            translation=trans_inv,
            translation_unit=self.translation_unit,
            extra=self.extra,
        )

    def transform_point(self, point: ArrayLike) -> ArrayLike:
        """Apply transform to point(s)."""
        rotated = matmul(self.rotation, point[..., None]).squeeze(-1)
        return rotated + self.translation

    def transform_vector(self, vector: ArrayLike) -> ArrayLike:
        """Apply rotation only to vector(s) (no translation)."""
        return matmul(self.rotation, vector[..., None]).squeeze(-1)

    def apply_delta(self, delta: "Transform", in_world_frame: bool = True) -> "Transform":
        """
        Apply a delta (incremental) transform.

        Args:
            delta: Incremental transform to apply
            in_world_frame: If True, delta is in world frame (result = delta @ self)
        """
        if in_world_frame:
            return delta @ self
        else:
            return self @ delta

    def relative_to(self, reference: "Transform") -> "Transform":
        """
        Compute this transform relative to a reference frame.

        Returns T such that: reference @ T = self
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
    def shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
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
        """Enable/disable gradient tracking in-place (PyTorch only)."""
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
        """Move transform to device/dtype (PyTorch only)."""
        if self.backend != "torch":
            raise ValueError("to() is only supported for PyTorch tensors")

        rot = self.rotation.to(device=device, dtype=dtype)
        trans = self.translation.to(device=device, dtype=dtype)
        extra = self.extra.to(device=device, dtype=dtype) if self.extra is not None else None
        return Transform(
            rotation=rot,
            translation=trans,
            translation_unit=self.translation_unit,
            extra=extra,
        )

    def to_unit(self, target_unit: Union[str, TranslationUnit]) -> "Transform":
        """Convert translation to a different unit."""
        if isinstance(target_unit, str):
            target_unit = TranslationUnit(target_unit)

        if target_unit == self.translation_unit:
            return self.clone()

        scale = UNIT_CONVERSION[(self.translation_unit, target_unit)]
        new_translation = self.translation * scale

        if self.backend == "torch":
            return Transform(
                rotation=self.rotation.clone(),
                translation=new_translation,
                translation_unit=target_unit,
                extra=self.extra.clone() if self.extra is not None else None,
            )
        return Transform(
            rotation=self.rotation.copy(),
            translation=new_translation,
            translation_unit=target_unit,
            extra=self.extra.copy() if self.extra is not None else None,
        )

    def detach(self) -> "Transform":
        """Detach from computation graph (PyTorch only)."""
        if self.backend != "torch":
            return self
        return Transform(
            rotation=self.rotation.detach(),
            translation=self.translation.detach(),
            translation_unit=self.translation_unit,
            extra=self.extra.detach() if self.extra is not None else None,
        )

    def clone(self) -> "Transform":
        """Create a copy of the transform."""
        if self.backend == "torch":
            return Transform(
                rotation=self.rotation.clone(),
                translation=self.translation.clone(),
                translation_unit=self.translation_unit,
                extra=self.extra.clone() if self.extra is not None else None,
            )
        return Transform(
            rotation=self.rotation.copy(),
            translation=self.translation.copy(),
            translation_unit=self.translation_unit,
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
            f"batch_shape={self.batch_shape}, "
            f"unit={self.translation_unit.value}{extra_info}, "
            f"dtype={self.dtype})"
        )

    def __getitem__(self, idx) -> "Transform":
        """Index into batched transforms."""
        return Transform(
            rotation=self.rotation[idx],
            translation=self.translation[idx],
            translation_unit=self.translation_unit,
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
        translation_unit: Union[str, TranslationUnit] = TranslationUnit.METER,
    ) -> ArrayLike:
        """Convert transform between representations without creating Transform instance."""
        transform = Transform.from_rep(
            tf,
            from_rep=from_rep,
            seq=seq,
            degrees=degrees,
            euler_in_rpy=euler_in_rpy,
            translation_unit=translation_unit,
        )
        return transform.to_rep(to_rep, seq=seq, degrees=degrees, euler_in_rpy=euler_in_rpy)

