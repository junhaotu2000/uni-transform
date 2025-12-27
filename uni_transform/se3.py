"""
SE(3) Lie group operations.

Provides logarithm and exponential maps between rigid transforms and twists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, overload

import numpy as np
import torch

from ._core import ArrayLike, EPS, SMALL_ANGLE_THRESHOLD, TranslationUnit, get_backend
from .rotation_conversions import matrix_to_rotvec, rotvec_to_matrix

if TYPE_CHECKING:
    from .transform import Transform


@overload
def se3_log(transform: "Transform") -> np.ndarray: ...
@overload
def se3_log(transform: "Transform") -> torch.Tensor: ...


def se3_log(transform: "Transform") -> ArrayLike:
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
    """
    backend = transform.backend

    omega = matrix_to_rotvec(transform.rotation)

    if backend == "numpy":
        angle = np.linalg.norm(omega, axis=-1, keepdims=True)
        small_angle = np.abs(angle) < SMALL_ANGLE_THRESHOLD

        if np.all(small_angle):
            v = transform.translation.copy()
        else:
            half_angle = angle / 2
            omega_cross_t = np.cross(omega, transform.translation, axis=-1)
            correction = (
                1 - angle * np.cos(half_angle) / (2 * np.sin(half_angle) + EPS)
            ) / (angle * angle + EPS)
            omega_cross_omega_cross_t = np.cross(omega, omega_cross_t, axis=-1)

            v = np.where(
                small_angle,
                transform.translation,
                transform.translation - 0.5 * omega_cross_t + correction * omega_cross_omega_cross_t,
            )

        return np.concatenate([omega, v], axis=-1)

    # PyTorch
    angle = torch.norm(omega, dim=-1, keepdim=True)
    small_angle = angle.abs() < SMALL_ANGLE_THRESHOLD

    half_angle = angle / 2
    omega_cross_t = torch.cross(omega, transform.translation, dim=-1)
    correction = (
        1 - angle * torch.cos(half_angle) / (2 * torch.sin(half_angle) + EPS)
    ) / (angle * angle + EPS)
    omega_cross_omega_cross_t = torch.cross(omega, omega_cross_t, dim=-1)

    v = torch.where(
        small_angle,
        transform.translation,
        transform.translation - 0.5 * omega_cross_t + correction * omega_cross_omega_cross_t,
    )

    return torch.cat([omega, v], dim=-1)


@overload
def se3_exp(
    twist: np.ndarray, translation_unit: Union[str, TranslationUnit] = ...
) -> "Transform": ...
@overload
def se3_exp(
    twist: torch.Tensor, translation_unit: Union[str, TranslationUnit] = ...
) -> "Transform": ...


def se3_exp(
    twist: ArrayLike,
    translation_unit: Union[str, TranslationUnit] = TranslationUnit.METER,
) -> "Transform":
    """
    Compute SE(3) exponential (twist to transform).

    Maps a 6D twist (se(3) Lie algebra element) to a rigid transform.

    Args:
        twist: 6D twist vector (..., 6) as [omega (3), v (3)]
        translation_unit: Unit of the translation component ("m" or "mm")

    Returns:
        Transform corresponding to the twist

    Note:
        This is the inverse of se3_log.
    """
    from .transform import Transform

    backend = get_backend(twist)

    omega = twist[..., :3]
    v = twist[..., 3:6]

    rotation = rotvec_to_matrix(omega)

    if backend == "numpy":
        angle = np.linalg.norm(omega, axis=-1, keepdims=True)
        small_angle = np.abs(angle) < SMALL_ANGLE_THRESHOLD

        if np.all(small_angle):
            translation = v.copy()
        else:
            omega_cross_v = np.cross(omega, v, axis=-1)
            omega_cross_omega_cross_v = np.cross(omega, omega_cross_v, axis=-1)

            c1 = (1 - np.cos(angle)) / (angle * angle + EPS)
            c2 = (angle - np.sin(angle)) / (angle * angle * angle + EPS)

            translation = np.where(
                small_angle,
                v,
                v + c1 * omega_cross_v + c2 * omega_cross_omega_cross_v,
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
            v + c1 * omega_cross_v + c2 * omega_cross_omega_cross_v,
        )

    return Transform(
        rotation=rotation,
        translation=translation,
        translation_unit=translation_unit,
    )

