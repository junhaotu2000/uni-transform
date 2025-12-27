# uni-transform

A Python library for 3D rigid body transformations with **NumPy** and **PyTorch** backends.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Key Features:**
- **Dual API** - `Transform` for poses (rotation + translation), `Rotation` for pure rotations
- **Batch Operations** - All functions support arbitrary batch dimensions `(..., N)`
- **PyTorch Gradients** - Fully differentiable for deep learning
- **Dual Backend** - Seamless NumPy ↔ PyTorch switching
- **Extra State** - Store gripper width, joint states alongside poses
- **Unit Support** - Explicit translation units (meters/millimeters)

## Installation

```bash
uv add uni-transform
```

From source:

```bash
git clone https://github.com/junhaotu/uni-transform.git
cd uni-transform
uv pip install -e .
```

## Quick Start

### Transform Class (Rotation + Translation)

```python
import numpy as np
from uni_transform import Transform

# Create from euler angles [x, y, z, roll, pitch, yaw]
tf = Transform.from_rep(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]), from_rep="euler")

# Convert representations
quat_rep = tf.to_rep("quat")      # [x, y, z, qx, qy, qz, qw]
matrix = tf.as_matrix()            # 4x4 homogeneous matrix

# Compose & transform
tf_composed = tf @ tf.inverse()
points = tf.transform_point(np.array([[1, 0, 0], [0, 1, 0]]))
```

### Rotation Class (Pure Rotation)

```python
from uni_transform import Rotation

# Create from various representations
rot = Rotation.from_euler(np.array([0.1, 0.2, 0.3]), seq="ZYX")
rot = Rotation.from_quat(np.array([0, 0, 0.707, 0.707]))
rot = Rotation.from_rotvec(np.array([0, 0, np.pi/2]))
rot = Rotation.from_rotation_6d(np.array([1, 0, 0, 0, 1, 0]))

# Or use generic from_rep
rot = Rotation.from_rep(np.array([0.1, 0.2, 0.3]), from_rep="euler", seq="ZYX")

# Convert to any representation
quat = rot.to_rep("quat")
euler = rot.as_euler(seq="ZYX", degrees=True)
matrix = rot.as_matrix()

# Compose rotations
combined = rot1 @ rot2

# Apply to vectors
rotated = rot.apply(np.array([1.0, 0.0, 0.0]))

# Interpolation
rot_mid = rot1.slerp(rot2, t=0.5)

# Relative rotation
rot_rel = rot2.relative_to(rot1)  # rot1 @ rot_rel = rot2
```

### With Extra State (Gripper Width, etc.)

```python
# Robot pose with gripper: [x, y, z, qx, qy, qz, qw, gripper_width]
pose = np.array([0.5, 0.2, 0.1, 0, 0, 0, 1, 0.04])
tf = Transform.from_rep(pose, from_rep="quat", extra_dims=1)

tf.translation  # [0.5, 0.2, 0.1]
tf.extra        # [0.04] - gripper width preserved

# Extra is included in conversions
euler_pose = tf.to_rep("euler")  # [x, y, z, r, p, y, gripper]

# Extra is interpolated too
tf_mid = transform_interpolate(tf_start, tf_end, t=0.5)
tf_mid.extra  # Linearly interpolated gripper width
```

### Translation Units

```python
# Explicit unit tracking prevents accidental mixing
tf_m = Transform.from_pos_quat([1.0, 2.0, 3.0], [0, 0, 0, 1], translation_unit="m")
tf_mm = tf_m.to_unit("mm")  # Translation becomes [1000, 2000, 3000]

# Unit mismatch raises error (catches bugs early)
tf_mm @ tf_m  # Raises UnitMismatchError

# Stack also requires matching units
Transform.stack([tf_m, tf_m.clone()])  # OK
Transform.stack([tf_m, tf_mm])         # Raises UnitMismatchError
```

### Rotation Conversions (Functional API)

```python
from uni_transform import quaternion_to_matrix, matrix_to_euler, convert_rotation

# Direct conversions
quat = np.array([0, 0, 0.707, 0.707])  # xyzw format
matrix = quaternion_to_matrix(quat)
euler = matrix_to_euler(matrix, seq="ZYX")

# Generic conversion
euler = convert_rotation(quat, from_rep="quat", to_rep="euler", seq="ZYX")

# Static method on classes (no instance needed)
euler = Rotation.convert(quat, from_rep="quat", to_rep="euler", seq="ZYX")
pose_quat = Transform.convert(pose_euler, from_rep="euler", to_rep="quat")
```

### PyTorch Gradients

```python
import torch
from uni_transform import Transform, Rotation, geodesic_distance

# Transform with gradients
pred = Transform.from_rep(torch.randn(100, 9), from_rep="rotation_6d", requires_grad=True)
target = Transform.from_rep(torch.randn(100, 9), from_rep="rotation_6d")

loss = geodesic_distance(pred.rotation, target.rotation).mean()
loss.backward()  # Fully differentiable

# Rotation with gradients
rot = Rotation.from_euler(torch.tensor([0.1, 0.2, 0.3], requires_grad=True))
rotated = rot.apply(torch.tensor([1.0, 0.0, 0.0]))
rotated.sum().backward()
```

### Batch Dimensions

```python
# All operations support arbitrary batch dimensions
batch_tf = Transform.from_rep(np.random.randn(10, 50, 7), from_rep="quat")  # (10, 50) batch
batch_tf.rotation.shape   # (10, 50, 3, 3)
batch_tf.translation.shape  # (10, 50, 3)

# Same for Rotation
batch_rot = Rotation.from_rep(np.random.randn(10, 50, 4), from_rep="quat")
batch_rot.batch_shape  # (10, 50)

# Compose batched transforms
result = batch_tf @ batch_tf.inverse()  # Broadcasting supported
```

### Interpolation

```python
from uni_transform import transform_interpolate, transform_sequence_interpolate, quaternion_slerp

# Interpolate between two transforms (includes extra interpolation)
tf_mid = transform_interpolate(tf_start, tf_end, t=0.5)

# Interpolate between two rotations
rot_mid = rot_start.slerp(rot_end, t=0.5)  # Spherical interpolation
rot_mid = rot_start.nlerp(rot_end, t=0.5)  # Faster, approximate

# Quaternion SLERP (functional)
q_mid = quaternion_slerp(q0, q1, t=0.5)

# Interpolate along a trajectory
keyframes = Transform.from_rep(np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0.5],
    [2, 0, 0, 0, 0, 1.0],
]), from_rep="euler")
times = np.array([0.0, 1.0, 2.0])
query_times = np.array([0.5, 1.5])
interpolated = transform_sequence_interpolate(keyframes, times, query_times)
```

### Relative Transforms & Deltas

```python
# Express tf in reference frame
tf_in_ref = tf.relative_to(reference_tf)  # = reference_tf.inverse() @ tf

# Apply incremental delta
tf_new = tf.apply_delta(delta_tf, in_world_frame=True)   # = delta_tf @ tf
tf_new = tf.apply_delta(delta_tf, in_world_frame=False)  # = tf @ delta_tf

# Same for Rotation
rot_rel = rot.relative_to(ref_rot)
rot_new = rot.apply_delta(delta_rot, in_body_frame=False)  # World frame
rot_new = rot.apply_delta(delta_rot, in_body_frame=True)   # Body frame
```

## Rotation Representations

| Name | Shape | Description |
|------|-------|-------------|
| `matrix` | `(..., 3, 3)` | SO(3) rotation matrix |
| `quat` | `(..., 4)` | Quaternion (**xyzw** format) |
| `euler` | `(..., 3)` | Euler angles (default: ZYX) |
| `rotation_6d` | `(..., 6)` | 6D continuous rotation |
| `rot_vec` | `(..., 3)` | Rotation vector (axis × angle) |

> `...` = arbitrary batch dimensions, e.g. `(B, T, 3, 3)` for batched trajectories

## Transform Class API

### Factory Methods

```python
Transform.identity(backend="numpy", extra_dims=0, translation_unit="m")
Transform.from_matrix(matrix_4x4, translation_unit="m")
Transform.from_rep(data, from_rep="quat", extra_dims=0, translation_unit="m")
Transform.from_pos_quat(position, quaternion, extra=None, translation_unit="m")
Transform.stack(transforms, axis=0)  # Requires matching units
Transform.convert(data, from_rep, to_rep, ...)  # Static conversion
```

### Properties

```python
tf.rotation          # (..., 3, 3) rotation matrix
tf.translation       # (..., 3) translation vector
tf.extra             # (..., K) extra state or None
tf.translation_unit  # TranslationUnit.METER or MILLIMETER
tf.backend           # "numpy" or "torch"
tf.batch_shape       # tuple of batch dimensions
tf.extra_dims        # number of extra dimensions (0 if None)
```

### Methods

```python
tf.as_matrix()           # 4x4 homogeneous matrix
tf.to_rep("quat")        # Convert to [translation, rotation, extra]
tf.to_unit("mm")         # Convert translation unit
tf.inverse()             # Inverse transform
tf.transform_point(p)    # Apply to point(s)
tf.transform_vector(v)   # Apply rotation only
tf.apply_delta(delta)    # Compose with delta
tf.relative_to(ref)      # Express in reference frame
tf.clone()               # Deep copy
tf.to(device, dtype)     # Move to device (PyTorch)
tf.detach()              # Detach from graph (PyTorch)
```

## Rotation Class API

### Factory Methods

```python
Rotation.identity(backend="numpy")
Rotation.from_matrix(matrix_3x3)
Rotation.from_rep(data, from_rep="quat", seq="ZYX", degrees=False)
Rotation.from_quat(quaternion)           # xyzw format
Rotation.from_euler(euler, seq="ZYX")
Rotation.from_rotvec(rotation_vector)
Rotation.from_rotation_6d(rot_6d)
Rotation.stack(rotations, axis=0)
Rotation.convert(data, from_rep, to_rep, ...)  # Static conversion
```

### Properties

```python
rot.matrix       # (..., 3, 3) rotation matrix
rot.backend      # "numpy" or "torch"
rot.batch_shape  # tuple of batch dimensions
rot.is_batched   # bool
```

### Methods

```python
rot.to_rep("euler", seq="ZYX")   # Convert to representation
rot.as_matrix()                   # (..., 3, 3)
rot.as_quat()                     # (..., 4) xyzw
rot.as_euler(seq="ZYX")           # (..., 3)
rot.as_rotvec()                   # (..., 3)
rot.as_rotation_6d()              # (..., 6)
rot.inverse()                     # Inverse rotation
rot.apply(vectors)                # Rotate vectors
rot.apply_delta(delta)            # Compose
rot.relative_to(ref)              # Relative rotation
rot.slerp(other, t)               # Spherical interpolation
rot.nlerp(other, t)               # Normalized linear interpolation
rot.clone()                       # Deep copy
rot.to(device, dtype)             # Move to device (PyTorch)
rot.detach()                      # Detach from graph (PyTorch)
```

## Key Functions

```python
# Conversions
quaternion_to_matrix, matrix_to_quaternion
euler_to_matrix, matrix_to_euler
rotvec_to_matrix, matrix_to_rotvec
rotation_6d_to_matrix, matrix_to_rotation_6d
convert_rotation, rotation_to_matrix, matrix_to_rotation

# Quaternion ops
quaternion_multiply, quaternion_apply, quaternion_inverse, quaternion_conjugate

# Distance (for loss functions)
geodesic_distance, translation_distance, transform_distance

# SE(3) Lie group
se3_log, se3_exp

# Interpolation
transform_interpolate, transform_sequence_interpolate
quaternion_slerp, quaternion_nlerp

# Utilities
orthogonalize_rotation, xyz_rotation_6d_to_matrix
```

## Conventions

- **Quaternion**: `xyzw` format (matches SciPy/ROS)
- **Euler default**: `ZYX` sequence (yaw-pitch-roll)
- **Composition**: `tf1 @ tf2` applies `tf2` first, then `tf1`
- **Translation unit**: Default is meters (`"m"`), also supports millimeters (`"mm"`)
- **Extra**: Preserved through `inverse()`, `clone()`, `to_unit()`; interpolated in `transform_interpolate()`

## Module Structure

```
uni_transform/
├── __init__.py              # Public API exports
├── _core.py                 # Types, constants, backend utilities
├── rotation_conversions.py  # All rotation conversion functions
├── rotation.py              # Rotation class
├── transform.py             # Transform class
├── interpolation.py         # Interpolation functions
├── se3.py                   # SE(3) Lie group operations
└── metrics.py               # Distance functions
```

## License

MIT License
