# uni-transform

A Python library for 3D rigid body transformations, supporting both **NumPy** and **PyTorch** backends with full gradient support.

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
uv add uni-transform
```

### From Source

```bash
git clone https://github.com/junhaotu/uni-transform.git
cd uni-transform
uv pip install -e .
```

### Development Installation

```bash
uv pip install -e ".[dev]"
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
from uni_transform import Transform, geodesic_distance, translation_distance, transform_distance

# Create transforms from 6D rotation representation (ideal for neural networks)
pred_traj = torch.randn(100, 9)  # [x, y, z, rot6d]
target_traj = torch.randn(100, 9)

pred = Transform.from_rep(pred_traj, from_rep="rotation_6d", requires_grad=True)
target = Transform.from_rep(target_traj, from_rep="rotation_6d")

# Option 1: Individual losses
rot_loss = geodesic_distance(pred.rotation, target.rotation, reduce=False)
trans_loss = translation_distance(pred.translation, target.translation, reduce=False)

# Option 2: Combined transform distance (returns tuple: total, rot, trans)
total_loss, rot_loss, trans_loss = transform_distance(pred, target, reduce=False)
loss = total_loss.mean()
loss.backward()  # Gradients flow through all operations
```

### Rotation Conversions

The library provides **two API styles** for rotation conversions, each suited to different use cases:

#### Style 1: Direct Functions (Recommended for Fixed Conversions)

Use when you know the exact conversion path at coding time. Benefits: type-safe, zero runtime overhead, clear function signatures.

```python
from uni_transform import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_to_matrix,
    matrix_to_euler,
    rotvec_to_matrix,
    rotation_6d_to_matrix,
)

# Direct, type-safe conversions
quat = np.array([0, 0, 0.707, 0.707])  # xyzw format, 90° around Z
matrix = quaternion_to_matrix(quat)
euler = matrix_to_euler(matrix, seq="ZYX")
rot6d = matrix_to_rotation_6d(matrix)
```

#### Style 2: Generic Functions (Recommended for Dynamic Conversions)

Use when the conversion type is determined at runtime (e.g., configurable pipelines). Benefits: flexible, fewer function imports.

```python
from uni_transform import convert_rotation, rotation_to_matrix, matrix_to_rotation

# Convert with runtime-determined representations
input_format = config.get("rotation_format")  # e.g., "quat", "euler"
output_format = "matrix"

matrix = convert_rotation(rotation_data, from_rep=input_format, to_rep=output_format)

# Or step-by-step via matrix as intermediate
matrix = rotation_to_matrix(rotation_data, from_rep=input_format)
result = matrix_to_rotation(matrix, to_rep=output_format)
```

#### When to Use Which?

| Scenario | Recommended Style |
|----------|-------------------|
| Fixed pipeline (e.g., always quat→matrix) | Direct functions |
| Neural network training loop | Direct functions (performance) |
| Config-driven conversion | Generic `convert_rotation` |
| Supporting multiple input formats | Generic `rotation_to_matrix` |
| One-off scripts | Either (personal preference) |

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

# Interpolate along a trajectory (batched transform input)
keyframes = np.array([
    [0, 0, 0, 0, 0, 0],  # [x, y, z, euler...]
    [1, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0],
])
transforms = Transform.from_rep(keyframes, from_rep="euler")  # Batched Transform
times = np.array([0.0, 1.0, 2.0])
query_times = np.array([0.5, 1.5])
interpolated = transform_sequence_interpolate(transforms, times, query_times)
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

#### Distance & Loss Functions

- `geodesic_distance(R1, R2)` - Rotation angle between matrices (radians)
- `translation_distance(t1, t2, p=2.0)` - Distance between translations (L1/L2 norm)
- `transform_distance(tf1, tf2)` - Combined rotation + translation distance

#### Utilities

- `orthogonalize_rotation(matrix)` - Project to SO(3) via SVD
- `se3_log(transform)` / `se3_exp(twist)` - SE(3) Lie group operations

### Transform Class

The `Transform` class provides an **object-oriented API** for rigid body transforms (rotation + translation). Use it when you need to:
- Compose multiple transforms
- Apply transforms to points/vectors
- Work with complete SE(3) poses

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

#### Transform vs Direct Functions

| Task | Transform Class | Direct Functions |
|------|-----------------|------------------|
| Rotation-only conversion | Overkill | ✅ Use `quaternion_to_matrix()` etc. |
| Pose with position + rotation | ✅ `Transform.from_rep()` | Manual concatenation |
| Composing transforms | ✅ `tf1 @ tf2` | Manual matrix multiply |
| Transforming points | ✅ `tf.transform_point()` | Manual `R @ p + t` |

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
