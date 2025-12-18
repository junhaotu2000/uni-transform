# uni-transform

A high-performance Python library for 3D rigid body transformations, supporting both **NumPy** and **PyTorch** backends with full gradient support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Dual Backend Support**: Seamless switching between NumPy and PyTorch
- **Full Gradient Support**: All PyTorch operations are differentiable (ideal for deep learning)
- **Arbitrary Batch Dimensions**: Handle single transforms or batches of any shape
- **Comprehensive Representations**: Matrix, Quaternion (xyzw), Euler, Rotation Vector, 6D Rotation
- **SE(3) Lie Group Operations**: Logarithm and exponential maps for twist representations
- **Interpolation**: SLERP, NLERP, and transform sequence interpolation
- **Numerically Stable**: Robust implementations handling edge cases (gimbal lock, 180° rotations)

## Installation

```bash
pip install uni-transform
```

### From Source

```bash
git clone https://github.com/junhaotu/uni-transform.git
cd uni-transform
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import numpy as np
from uni_transform import Transform, convert_rotation

# Create a transform from position + euler angles
tf = Transform.from_rep(
    np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),  # [x, y, z, roll, pitch, yaw]
    from_rep="euler",
    seq="ZYX"
)

# Convert to quaternion representation
quat_rep = tf.to_rep("quat")  # [x, y, z, qx, qy, qz, qw]

# Get 4x4 homogeneous matrix
matrix_4x4 = tf.as_matrix()

# Transform composition
tf_composed = tf @ tf.inverse()  # Should be identity

# Transform points
points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
transformed = tf.transform_point(points)
```

### PyTorch with Gradients

```python
import torch
from uni_transform import rotation_6d_to_matrix, geodesic_distance

# 6D rotation is ideal for neural networks (continuous, no singularities)
pred_rot6d = model(input)  # Shape: (batch, 6)

# Convert to rotation matrix
pred_matrix = rotation_6d_to_matrix(pred_rot6d)  # Shape: (batch, 3, 3)

# Compute geodesic loss (gradients flow through)
loss = geodesic_distance(pred_matrix, target_matrix, reduce=False).mean()
loss.backward()
```

### Rotation Conversions

```python
from uni_transform import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_to_matrix,
    matrix_to_euler,
    rotvec_to_matrix,
    convert_rotation,
)

# Direct conversions
quat = np.array([0, 0, 0.707, 0.707])  # xyzw format, 90° around Z
matrix = quaternion_to_matrix(quat)

# Generic conversion function
euler = convert_rotation(quat, from_rep="quat", to_rep="euler", seq="ZYX")
rot6d = convert_rotation(matrix, from_rep="matrix", to_rep="rotation_6d")
```

### Quaternion Operations

```python
from uni_transform import (
    quaternion_multiply,
    quaternion_apply,
    quaternion_slerp,
    quaternion_inverse,
)

# Quaternion multiplication (composition)
q_combined = quaternion_multiply(q1, q2)

# Rotate a vector directly (more efficient than matrix multiplication)
rotated_vector = quaternion_apply(quat, vector)

# Spherical interpolation
q_mid = quaternion_slerp(q_start, q_end, t=0.5)
```

### Transform Interpolation

```python
from uni_transform import transform_interpolate, transform_sequence_interpolate

# Interpolate between two transforms
tf_mid = transform_interpolate(tf_start, tf_end, t=0.5)

# Interpolate along a trajectory
keyframes = [tf0, tf1, tf2, tf3]
times = np.array([0.0, 1.0, 2.0, 3.0])
query_times = np.array([0.5, 1.5, 2.5])
interpolated = transform_sequence_interpolate(keyframes, times, query_times)
```

### SE(3) Lie Group Operations

```python
from uni_transform import se3_log, se3_exp

# Convert transform to twist (6D tangent space)
twist = se3_log(transform)  # [omega_x, omega_y, omega_z, v_x, v_y, v_z]

# Convert twist back to transform
transform_recovered = se3_exp(twist)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Transform` | Rigid body transform with rotation matrix and translation |
| `RotationRepr` | Enum for rotation representations: `EULER`, `QUAT`, `MATRIX`, `ROTATION_6D`, `ROT_VEC` |

### Rotation Representations

| Representation | Shape | Description |
|----------------|-------|-------------|
| `matrix` | `(..., 3, 3)` | Rotation matrix (SO(3)) |
| `quat` | `(..., 4)` | Quaternion in **xyzw** format |
| `euler` | `(..., 3)` | Euler angles (default: ZYX sequence) |
| `rotation_6d` | `(..., 6)` | 6D rotation (first two rows of matrix) |
| `rot_vec` | `(..., 3)` | Rotation vector (axis × angle) |

### Key Functions

#### Conversion Functions

- `quaternion_to_matrix(quat)` / `matrix_to_quaternion(matrix)`
- `euler_to_matrix(euler, seq="ZYX")` / `matrix_to_euler(matrix, seq="ZYX")`
- `rotvec_to_matrix(rotvec)` / `matrix_to_rotvec(matrix)`
- `rotation_6d_to_matrix(rot6d)` / `matrix_to_rotation_6d(matrix)`
- `convert_rotation(rotation, from_rep=..., to_rep=...)`

#### Quaternion Operations

- `quaternion_multiply(q1, q2)` - Hamilton product
- `quaternion_apply(q, v)` - Rotate vector by quaternion
- `quaternion_conjugate(q)` / `quaternion_inverse(q)`
- `quaternion_slerp(q0, q1, t)` - Spherical linear interpolation
- `quaternion_nlerp(q0, q1, t)` - Normalized linear interpolation

#### Utilities

- `geodesic_distance(R1, R2)` - Rotation angle between matrices
- `orthogonalize_rotation(matrix)` - Project to SO(3) via SVD
- `se3_log(transform)` / `se3_exp(twist)` - SE(3) Lie group operations

### Transform Class Methods

```python
# Factory methods
Transform.identity(backend="numpy")
Transform.from_matrix(matrix_4x4)
Transform.from_rep(array, from_rep="euler")
Transform.from_pos_quat(position, quaternion)
Transform.stack([tf1, tf2, tf3])

# Instance methods
tf.as_matrix()              # Get 4x4 matrix
tf.to_rep("quat")           # Convert to representation
tf.inverse()                # Inverse transform
tf.transform_point(point)   # Apply to points
tf.transform_vector(vector) # Apply rotation only
tf.apply_delta(delta)       # Apply incremental transform
tf.relative_to(reference)   # Express in reference frame
tf.clone()                  # Deep copy
tf.to(device="cuda")        # Move to device (PyTorch)
tf.requires_grad_(True)     # Enable gradients (PyTorch)
```

## Conventions

- **Quaternion format**: `xyzw` (matches SciPy and ROS conventions)
- **Euler default**: `ZYX` sequence (yaw-pitch-roll)
- **Transform composition**: `tf1 @ tf2` applies `tf2` first, then `tf1`

## Performance Tips

1. **For neural networks**: Use `rotation_6d` representation (continuous, no singularities)
2. **For single vector rotations**: Use `quaternion_apply` instead of matrix multiplication
3. **For many interpolations**: Use `quaternion_nlerp` instead of `quaternion_slerp`
4. **For trajectory interpolation**: Use vectorized `transform_sequence_interpolate`

## Testing

```bash
pytest test/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# uni-transform
