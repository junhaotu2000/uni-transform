"""
uni_transform - Unified transform utilities supporting both NumPy and PyTorch backends.

A high-performance library for 3D rigid body transformations with:
- Consistent Gram-Schmidt implementation across backends
- Unified quaternion convention (xyzw throughout)
- Full PyTorch gradient support for differentiable robotics
- Arbitrary batch dimension support

API Styles
----------
Two styles for rotation conversions:

1. Direct Functions - for fixed, known conversions (type-safe, fast):
    >>> matrix = quaternion_to_matrix(quat)
    >>> euler = matrix_to_euler(matrix, seq="ZYX")

2. Generic Functions - for dynamic/configurable conversions:
    >>> matrix = convert_rotation(data, from_rep="quat", to_rep="matrix")
    >>> euler = convert_rotation(data, from_rep=config["input_format"], to_rep="euler")

3. Transform Class - for full poses (rotation + translation):
    >>> tf = Transform.from_rep(np.array([x, y, z, r, p, y]), from_rep="euler")
    >>> quat_rep = tf.to_rep("quat")  # [x, y, z, qx, qy, qz, qw]

See README.md for detailed guidance on when to use each style.
"""

__version__ = "0.1.0"

from .main import (
    # Core types & constants
    ArrayLike,
    Backend,
    RotationRepr,
    Transform,
    EPS,
    SMALL_ANGLE_THRESHOLD,
    # 6D rotation
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    # Quaternion (xyzw)
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_apply,
    # Euler
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_euler_differentiable,
    # Rotation vector
    rotvec_to_matrix,
    matrix_to_rotvec,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    # Generic conversion
    rotation_to_matrix,
    matrix_to_rotation,
    convert_rotation,
    # Interpolation
    quaternion_slerp,
    quaternion_nlerp,
    transform_interpolate,
    transform_sequence_interpolate,
    # SE(3) Lie group operations
    se3_log,
    se3_exp,
    # Utilities
    xyz_rotation_6d_to_matrix,
    geodesic_distance,
    orthogonalize_rotation,
)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

def rotation_from_rep(rotation, from_rep, **kwargs):
    """Deprecated: Use rotation_to_matrix instead."""
    return rotation_to_matrix(rotation, from_rep, **kwargs)


def rotation_to_rep(matrix, to_rep, **kwargs):
    """Deprecated: Use matrix_to_rotation instead."""
    return matrix_to_rotation(matrix, to_rep, **kwargs)


def mtx_to_rot_6d(matrix):
    """Deprecated: Use matrix_to_rotation_6d instead."""
    return matrix_to_rotation_6d(matrix)


def rot_6d_to_mtx(rot_6d):
    """Deprecated: Use rotation_6d_to_matrix instead."""
    return rotation_6d_to_matrix(rot_6d)


def convert_transform(tf, *, from_rep, to_rep, **kwargs):
    """Deprecated: Use Transform.convert instead."""
    return Transform.convert(tf, from_rep=from_rep, to_rep=to_rep, **kwargs)


def geodesic_loss(R1, R2, reduce=True, degrees=False):
    """Deprecated: Use geodesic_distance instead."""
    return geodesic_distance(R1, R2, reduce=reduce, degrees=degrees)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core types & constants
    "ArrayLike",
    "Backend",
    "RotationRepr",
    "Transform",
    "EPS",
    "SMALL_ANGLE_THRESHOLD",
    # 6D rotation
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    # Quaternion (xyzw)
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_apply",
    # Euler
    "euler_to_matrix",
    "matrix_to_euler",
    "matrix_to_euler_differentiable",
    # Rotation vector
    "rotvec_to_matrix",
    "matrix_to_rotvec",
    "quaternion_to_rotvec",
    "rotvec_to_quaternion",
    # Generic conversion
    "rotation_to_matrix",
    "matrix_to_rotation",
    "convert_rotation",
    # Interpolation
    "quaternion_slerp",
    "quaternion_nlerp",
    "transform_interpolate",
    "transform_sequence_interpolate",
    # SE(3) Lie group operations
    "se3_log",
    "se3_exp",
    # Utilities
    "xyz_rotation_6d_to_matrix",
    "geodesic_distance",
    "orthogonalize_rotation",
    # Backward compatibility
    "rotation_from_rep",
    "rotation_to_rep",
    "mtx_to_rot_6d",
    "rot_6d_to_mtx",
    "convert_transform",
    "geodesic_loss",
]
