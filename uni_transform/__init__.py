"""
Unified transform utilities supporting both NumPy and PyTorch backends.

API Styles
----------
This library provides three API styles:

1. Direct Functions (recommended for fixed conversions):
   - Type-safe, zero runtime overhead, clear signatures
   - Examples: quaternion_to_matrix(), euler_to_matrix(), matrix_to_rotvec()

2. Generic Functions (recommended for dynamic/configurable conversions):
   - Flexible, representation determined at runtime
   - Examples: convert_rotation(), rotation_to_matrix(), matrix_to_rotation()

3. Class-based API (recommended for object-oriented usage):
   - Rotation class: for rotations only
   - Transform class: for full poses (rotation + translation)
   - Unified from_rep/to_rep interface

Usage Examples
--------------
Direct rotation conversion (fixed path):
    matrix = quaternion_to_matrix(quat)
    euler = matrix_to_euler(matrix, seq="ZYX")

Generic rotation conversion (dynamic path):
    matrix = convert_rotation(rotation, from_rep=input_format, to_rep="matrix")

Rotation class (for rotations only):
    rot = Rotation.from_rep(quat, from_rep="quat")
    euler = rot.to_rep("euler", seq="ZYX")
    rot_composed = rot1 @ rot2
    rotated_v = rot.apply(v)

Transform class (for poses):
    tf = Transform.from_rep(
        np.array([x, y, z, roll, pitch, yaw]),
        from_rep="euler",
        seq="ZYX",
    )
    quat_rep = tf.to_rep("quat")  # [x, y, z, qx, qy, qz, qw]

Conventions
-----------
- Quaternion: xyzw (matches SciPy/ROS convention)
- Euler: default seq="ZYX", can be overridden
- Rotation matrix: (..., 3, 3)
- Transform matrix: (..., 4, 4), rotation in [:3, :3], translation in [:3, 3]
"""

# Types and constants
from ._core import (
    ArrayLike,
    Backend,
    EPS,
    RotationRepr,
    SMALL_ANGLE_THRESHOLD,
    TranslationUnit,
    UnitMismatchError,
)

# Rotation conversion functions
from .rotation_conversions import (
    # 6D rotation
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    # Quaternion
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_apply,
    # Euler angles
    euler_to_matrix,
    matrix_to_euler,
    matrix_to_euler_differentiable,
    # Rotation vector (axis-angle)
    rotvec_to_matrix,
    matrix_to_rotvec,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    # Generic conversion
    rotation_to_matrix,
    matrix_to_rotation,
    convert_rotation,
    # Utilities
    orthogonalize_rotation,
    xyz_rotation_6d_to_matrix,
)

# Classes
from .rotation import Rotation
from .transform import Transform

# Interpolation
from .interpolation import (
    quaternion_slerp,
    quaternion_nlerp,
    transform_interpolate,
    transform_sequence_interpolate,
)

# SE(3) Lie group
from .se3 import se3_log, se3_exp

# Metrics
from .metrics import (
    geodesic_distance,
    translation_distance,
    transform_distance,
)

__all__ = [
    # Types
    "ArrayLike",
    "Backend",
    "RotationRepr",
    "TranslationUnit",
    # Constants
    "EPS",
    "SMALL_ANGLE_THRESHOLD",
    # Classes
    "Rotation",
    "Transform",
    "UnitMismatchError",
    # 6D rotation
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    # Quaternion
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
    # SE(3)
    "se3_log",
    "se3_exp",
    # Metrics
    "geodesic_distance",
    "translation_distance",
    "transform_distance",
    "orthogonalize_rotation",
    "xyz_rotation_6d_to_matrix",
]

__version__ = "0.3.0"
