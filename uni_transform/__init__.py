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
    QuatInterpMethod,
    RotationRepr,
    RotationSeqInterpMethod,
    SMALL_ANGLE_THRESHOLD,
    TranslationUnit,
    UnitMismatchError,
    VectorInterpMethod,
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
    # interpolation functions
    interpolate,
    interpolate_sequence,
    interpolate_rotation,
    interpolate_rotation_sequence,
    interpolate_transform,
    interpolate_transform_sequence,
    # spline functions
    SplineCoefficients,
    compute_spline,
    # quaternion functions
    quaternion_slerp,
    quaternion_nlerp,
    quaternion_squad,
    minimum_jerk_interpolate,
    minimum_jerk_velocity,
    minimum_jerk_acceleration,
    cubic_spline_coefficients,
    cubic_spline_interpolate,
    cubic_spline_derivative,
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
    "VectorInterpMethod",
    "QuatInterpMethod",
    "RotationSeqInterpMethod",
    # Constants
    "EPS",
    "SMALL_ANGLE_THRESHOLD",
    # Classes
    "Rotation",
    "Transform",
    "UnitMismatchError",
    # Interpolation - Unified API
    "interpolate",
    "interpolate_sequence",
    "interpolate_rotation",
    "interpolate_rotation_sequence",
    "interpolate_transform",
    "interpolate_transform_sequence",
    # Interpolation - Helpers
    "SplineCoefficients",
    "compute_spline",
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
    # Interpolation - Quaternion
    "quaternion_slerp",
    "quaternion_nlerp",
    "quaternion_squad",
    # Interpolation - Minimum Jerk
    "minimum_jerk_interpolate",
    "minimum_jerk_velocity",
    "minimum_jerk_acceleration",
    # Interpolation - Cubic Spline
    "compute_spline",
    "cubic_spline_coefficients",
    "cubic_spline_interpolate",
    "cubic_spline_derivative",
    # Interpolation - Transform
    "interpolate_transform",
    "interpolate_transform_sequence",
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

__version__ = "0.4.0"
