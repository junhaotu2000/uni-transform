"""
Comprehensive tests for uni_transform package.

Test categories:
1. NumPy/PyTorch consistency
2. Roundtrip conversions
3. Edge cases (180-degree rotations, zero rotations, gimbal lock)
4. Gradient flow
5. Interpolation
6. Batch operations
7. Transform class methods

Run with:
    pytest test/test.py -v
    # or from project root:
    pytest -v
"""

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch

from uni_transform import (
    # Core
    Transform,
    RotationRepr,
    # 6D rotation
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    # Quaternion
    quaternion_to_matrix,
    matrix_to_quaternion,
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
    # Quaternion operations
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_apply,
    # Interpolation
    quaternion_slerp,
    quaternion_nlerp,
    transform_interpolate,
    transform_sequence_interpolate,
    # SE(3) Lie group
    se3_log,
    se3_exp,
    # Utilities
    geodesic_distance,
    orthogonalize_rotation,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def random_rotation_matrices_np():
    """Generate random valid rotation matrices (NumPy)."""
    np.random.seed(42)
    # Generate random axis-angle and convert to matrices
    rotvecs = np.random.randn(10, 3)
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rotvecs).as_matrix()


@pytest.fixture
def random_rotation_matrices_torch(random_rotation_matrices_np):
    """Generate random valid rotation matrices (PyTorch)."""
    return torch.from_numpy(random_rotation_matrices_np).float()


@pytest.fixture
def random_quaternions_np():
    """Generate random unit quaternions (xyzw format)."""
    np.random.seed(42)
    quats = np.random.randn(10, 4)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    # Ensure positive w (canonical form)
    quats = np.where(quats[..., 3:4] < 0, -quats, quats)
    return quats


@pytest.fixture
def random_quaternions_torch(random_quaternions_np):
    """Generate random unit quaternions (PyTorch)."""
    return torch.from_numpy(random_quaternions_np).float()


@pytest.fixture
def identity_matrix_np():
    """Identity rotation matrix (NumPy)."""
    return np.eye(3)


@pytest.fixture
def identity_matrix_torch():
    """Identity rotation matrix (PyTorch)."""
    return torch.eye(3)


# =============================================================================
# Test: NumPy/PyTorch Consistency
# =============================================================================


class TestNumpyPyTorchConsistency:
    """Verify that NumPy and PyTorch implementations produce identical results."""

    def test_rotation_6d_to_matrix_consistency(self, random_rotation_matrices_np):
        """6D rotation to matrix should be consistent across backends."""
        rot_6d_np = matrix_to_rotation_6d(random_rotation_matrices_np)
        rot_6d_torch = matrix_to_rotation_6d(torch.from_numpy(random_rotation_matrices_np).float())
        
        matrix_np = rotation_6d_to_matrix(rot_6d_np)
        matrix_torch = rotation_6d_to_matrix(rot_6d_torch).numpy()
        
        np.testing.assert_allclose(matrix_np, matrix_torch, rtol=1e-5, atol=1e-6)

    def test_quaternion_to_matrix_consistency(self, random_quaternions_np):
        """Quaternion to matrix should be consistent across backends."""
        matrix_np = quaternion_to_matrix(random_quaternions_np)
        matrix_torch = quaternion_to_matrix(
            torch.from_numpy(random_quaternions_np).float()
        ).numpy()
        
        np.testing.assert_allclose(matrix_np, matrix_torch, rtol=1e-5, atol=1e-6)

    def test_matrix_to_quaternion_consistency(self, random_rotation_matrices_np):
        """Matrix to quaternion should be consistent across backends."""
        quat_np = matrix_to_quaternion(random_rotation_matrices_np)
        quat_torch = matrix_to_quaternion(
            torch.from_numpy(random_rotation_matrices_np).float()
        ).numpy()
        
        # Quaternions can differ by sign, so compare absolute values or rotation matrices
        matrix_from_np = quaternion_to_matrix(quat_np)
        matrix_from_torch = quaternion_to_matrix(quat_torch)
        
        np.testing.assert_allclose(matrix_from_np, matrix_from_torch, rtol=1e-5, atol=1e-6)

    def test_euler_to_matrix_consistency(self):
        """Euler to matrix should be consistent across backends."""
        euler_angles = np.array([[0.1, 0.2, 0.3], [0.5, -0.3, 0.8], [1.0, 0.0, -0.5]])
        
        for seq in ["ZYX", "XYZ", "YZX"]:
            matrix_np = euler_to_matrix(euler_angles, seq=seq)
            matrix_torch = euler_to_matrix(
                torch.from_numpy(euler_angles).float(), seq=seq
            ).numpy()
            
            np.testing.assert_allclose(
                matrix_np, matrix_torch, rtol=1e-5, atol=1e-6,
                err_msg=f"Mismatch for seq={seq}"
            )

    def test_rotvec_to_matrix_consistency(self):
        """Rotation vector to matrix should be consistent across backends."""
        rotvecs = np.array([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        matrix_np = rotvec_to_matrix(rotvecs)
        matrix_torch = rotvec_to_matrix(torch.from_numpy(rotvecs).float()).numpy()
        
        np.testing.assert_allclose(matrix_np, matrix_torch, rtol=1e-5, atol=1e-6)

    def test_quaternion_slerp_consistency(self, random_quaternions_np):
        """SLERP should be consistent across backends."""
        q0 = random_quaternions_np[0]
        q1 = random_quaternions_np[1]
        t = 0.3
        
        result_np = quaternion_slerp(q0, q1, t)
        result_torch = quaternion_slerp(
            torch.from_numpy(q0).float(),
            torch.from_numpy(q1).float(),
            t
        ).numpy()
        
        np.testing.assert_allclose(result_np, result_torch, rtol=1e-5, atol=1e-6)


# =============================================================================
# Test: Roundtrip Conversions
# =============================================================================


class TestRoundtripConversions:
    """Verify that A -> B -> A == A for all representations."""

    def test_matrix_6d_roundtrip(self, random_rotation_matrices_np):
        """Matrix -> 6D -> Matrix should preserve the original."""
        rot_6d = matrix_to_rotation_6d(random_rotation_matrices_np)
        recovered = rotation_6d_to_matrix(rot_6d)
        
        np.testing.assert_allclose(
            random_rotation_matrices_np, recovered, rtol=1e-5, atol=1e-6
        )

    def test_matrix_quaternion_roundtrip(self, random_rotation_matrices_np):
        """Matrix -> Quaternion -> Matrix should preserve the original."""
        quat = matrix_to_quaternion(random_rotation_matrices_np)
        recovered = quaternion_to_matrix(quat)
        
        np.testing.assert_allclose(
            random_rotation_matrices_np, recovered, rtol=1e-5, atol=1e-6
        )

    def test_matrix_euler_roundtrip(self, random_rotation_matrices_np):
        """Matrix -> Euler -> Matrix should preserve the original."""
        for seq in ["ZYX", "XYZ"]:
            euler = matrix_to_euler(random_rotation_matrices_np, seq=seq)
            recovered = euler_to_matrix(euler, seq=seq)
            
            np.testing.assert_allclose(
                random_rotation_matrices_np, recovered, rtol=1e-4, atol=1e-5,
                err_msg=f"Roundtrip failed for seq={seq}"
            )

    def test_matrix_rotvec_roundtrip(self, random_rotation_matrices_np):
        """Matrix -> RotVec -> Matrix should preserve the original."""
        rotvec = matrix_to_rotvec(random_rotation_matrices_np)
        recovered = rotvec_to_matrix(rotvec)
        
        np.testing.assert_allclose(
            random_rotation_matrices_np, recovered, rtol=1e-5, atol=1e-6
        )

    def test_quaternion_rotvec_roundtrip(self, random_quaternions_np):
        """Quaternion -> RotVec -> Quaternion should preserve the original."""
        rotvec = quaternion_to_rotvec(random_quaternions_np)
        recovered = rotvec_to_quaternion(rotvec)
        
        # Quaternions can differ by sign
        dot = np.abs(np.sum(random_quaternions_np * recovered, axis=-1))
        np.testing.assert_allclose(dot, 1.0, rtol=1e-5, atol=1e-6)

    def test_convert_rotation_roundtrip(self, random_rotation_matrices_np):
        """convert_rotation roundtrip through all representations."""
        representations = ["quat", "euler", "rotation_6d", "rot_vec"]
        
        for rep in representations:
            converted = convert_rotation(
                random_rotation_matrices_np, from_rep="matrix", to_rep=rep
            )
            recovered = convert_rotation(converted, from_rep=rep, to_rep="matrix")
            
            np.testing.assert_allclose(
                random_rotation_matrices_np, recovered, rtol=1e-4, atol=1e-5,
                err_msg=f"Roundtrip failed for {rep}"
            )


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test numerical edge cases."""

    def test_identity_rotation(self):
        """Identity rotation should work correctly."""
        identity = np.eye(3)
        
        # To quaternion
        quat = matrix_to_quaternion(identity)
        expected_quat = np.array([0, 0, 0, 1])
        np.testing.assert_allclose(quat, expected_quat, rtol=1e-6, atol=1e-6)
        
        # To euler
        euler = matrix_to_euler(identity, seq="ZYX")
        expected_euler = np.array([0, 0, 0])
        np.testing.assert_allclose(euler, expected_euler, rtol=1e-6, atol=1e-6)
        
        # To rotvec
        rotvec = matrix_to_rotvec(identity)
        expected_rotvec = np.array([0, 0, 0])
        np.testing.assert_allclose(rotvec, expected_rotvec, rtol=1e-6, atol=1e-6)

    def test_180_degree_rotation_x(self):
        """180-degree rotation around X axis."""
        # R_x(pi)
        matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float64)
        
        # Should convert and recover
        quat = matrix_to_quaternion(matrix)
        recovered = quaternion_to_matrix(quat)
        
        np.testing.assert_allclose(matrix, recovered, rtol=1e-5, atol=1e-6)

    def test_180_degree_rotation_z(self):
        """180-degree rotation around Z axis."""
        matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Should convert and recover
        rot_6d = matrix_to_rotation_6d(matrix)
        recovered = rotation_6d_to_matrix(rot_6d)
        
        np.testing.assert_allclose(matrix, recovered, rtol=1e-5, atol=1e-6)

    def test_near_gimbal_lock(self):
        """Test behavior near gimbal lock (pitch ≈ ±90 degrees)."""
        # Near gimbal lock for ZYX: pitch close to 90 degrees
        euler = np.array([0.1, np.pi/2 - 0.01, 0.2])
        
        matrix = euler_to_matrix(euler, seq="ZYX")
        recovered_euler = matrix_to_euler(matrix, seq="ZYX")
        recovered_matrix = euler_to_matrix(recovered_euler, seq="ZYX")
        
        # The matrices should match even if euler angles differ
        np.testing.assert_allclose(matrix, recovered_matrix, rtol=1e-4, atol=1e-5)

    def test_zero_rotation_vector(self):
        """Zero rotation vector should give identity."""
        rotvec = np.array([0, 0, 0])
        matrix = rotvec_to_matrix(rotvec)
        
        np.testing.assert_allclose(matrix, np.eye(3), rtol=1e-6, atol=1e-6)

    def test_very_small_rotation(self):
        """Very small rotation should work correctly."""
        rotvec = np.array([1e-10, 1e-10, 1e-10])
        
        matrix = rotvec_to_matrix(rotvec)
        recovered = matrix_to_rotvec(matrix)
        
        # Should be close to original
        np.testing.assert_allclose(rotvec, recovered, rtol=1e-5, atol=1e-8)


# =============================================================================
# Test: Gradient Flow
# =============================================================================


class TestGradientFlow:
    """Test that gradients flow correctly through PyTorch operations."""

    def test_rotation_6d_gradient(self):
        """Gradients should flow through 6D rotation conversion."""
        rot_6d = torch.randn(4, 6, requires_grad=True)
        
        matrix = rotation_6d_to_matrix(rot_6d)
        loss = matrix.sum()
        loss.backward()
        
        assert rot_6d.grad is not None
        assert not torch.isnan(rot_6d.grad).any()

    def test_quaternion_to_matrix_gradient(self):
        """Gradients should flow through quaternion to matrix."""
        # Use leaf tensor for gradient tracking
        quat_raw = torch.randn(4, 4, requires_grad=True)
        
        # Normalize without breaking gradient flow
        quat = quat_raw / quat_raw.norm(dim=-1, keepdim=True)
        
        matrix = quaternion_to_matrix(quat)
        loss = matrix.sum()
        loss.backward()
        
        # Check gradient on the leaf tensor
        assert quat_raw.grad is not None
        assert not torch.isnan(quat_raw.grad).any()

    def test_euler_to_matrix_gradient(self):
        """Gradients should flow through euler to matrix."""
        euler = torch.randn(4, 3, requires_grad=True)
        
        matrix = euler_to_matrix(euler, seq="ZYX")
        loss = matrix.sum()
        loss.backward()
        
        assert euler.grad is not None
        assert not torch.isnan(euler.grad).any()

    def test_matrix_to_euler_differentiable_gradient(self):
        """Gradients should flow through differentiable euler extraction."""
        # Start with euler, convert to matrix, then back
        euler = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        
        matrix = euler_to_matrix(euler, seq="ZYX")
        recovered = matrix_to_euler_differentiable(matrix, seq="ZYX")
        loss = recovered.sum()
        loss.backward()
        
        assert euler.grad is not None
        assert not torch.isnan(euler.grad).any()

    def test_rotvec_to_matrix_gradient(self):
        """Gradients should flow through rotvec to matrix."""
        rotvec = torch.randn(4, 3, requires_grad=True)
        
        matrix = rotvec_to_matrix(rotvec)
        loss = matrix.sum()
        loss.backward()
        
        assert rotvec.grad is not None
        assert not torch.isnan(rotvec.grad).any()

    def test_quaternion_slerp_gradient(self):
        """Gradients should flow through SLERP."""
        q0 = torch.randn(4, requires_grad=True)
        q0_norm = q0 / q0.norm()
        q1 = torch.randn(4, requires_grad=True)
        q1_norm = q1 / q1.norm()
        t = 0.5
        
        result = quaternion_slerp(q0_norm, q1_norm, t)
        loss = result.sum()
        loss.backward()
        
        assert q0.grad is not None
        assert q1.grad is not None

    def test_geodesic_distance_gradient(self):
        """Gradients should flow through geodesic distance."""
        R1 = torch.eye(3, requires_grad=True)
        R2 = torch.randn(3, 3)
        R2 = orthogonalize_rotation(R2)
        
        dist = geodesic_distance(R1, R2, reduce=False)
        loss = dist.sum()
        loss.backward()
        
        assert R1.grad is not None


# =============================================================================
# Test: Interpolation
# =============================================================================


class TestInterpolation:
    """Test interpolation functions."""

    def test_slerp_endpoints(self, random_quaternions_np):
        """SLERP at t=0 and t=1 should return endpoints."""
        q0 = random_quaternions_np[0]
        q1 = random_quaternions_np[1]
        
        result_0 = quaternion_slerp(q0, q1, 0.0)
        result_1 = quaternion_slerp(q0, q1, 1.0)
        
        np.testing.assert_allclose(result_0, q0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result_1, q1, rtol=1e-6, atol=1e-6)

    def test_slerp_midpoint(self):
        """SLERP at t=0.5 should be halfway between rotations."""
        # Identity to 90 degrees around Z
        q0 = np.array([0, 0, 0, 1])
        q1 = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])  # 90 deg
        
        result = quaternion_slerp(q0, q1, 0.5)
        
        # Should be 45 degrees around Z
        expected = np.array([0, 0, np.sin(np.pi/8), np.cos(np.pi/8)])
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_nlerp_endpoints(self, random_quaternions_np):
        """NLERP at t=0 and t=1 should return endpoints."""
        q0 = random_quaternions_np[0]
        q1 = random_quaternions_np[1]
        
        result_0 = quaternion_nlerp(q0, q1, 0.0)
        result_1 = quaternion_nlerp(q0, q1, 1.0)
        
        np.testing.assert_allclose(result_0, q0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result_1, q1, rtol=1e-6, atol=1e-6)

    def test_slerp_vs_nlerp_small_angle(self):
        """SLERP and NLERP should be similar for small angles."""
        q0 = np.array([0, 0, 0, 1])
        q1 = np.array([0, 0, np.sin(0.05), np.cos(0.05)])  # ~5.7 deg
        
        for t in [0.25, 0.5, 0.75]:
            slerp_result = quaternion_slerp(q0, q1, t)
            nlerp_result = quaternion_nlerp(q0, q1, t)
            
            np.testing.assert_allclose(slerp_result, nlerp_result, rtol=1e-3, atol=1e-4)

    def test_slerp_batched(self, random_quaternions_np):
        """SLERP should work with batched inputs."""
        q0 = random_quaternions_np[:5]
        q1 = random_quaternions_np[5:]
        t = 0.3
        
        result = quaternion_slerp(q0, q1, t)
        
        assert result.shape == (5, 4)
        # Each result should be unit quaternion
        norms = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_transform_interpolate_endpoints(self):
        """Transform interpolation at t=0 and t=1 should return endpoints."""
        tf0 = Transform.identity(backend="numpy")
        tf1 = Transform.from_rep(
            np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
            from_rep="euler"
        )
        
        result_0 = transform_interpolate(tf0, tf1, 0.0)
        result_1 = transform_interpolate(tf0, tf1, 1.0)
        
        np.testing.assert_allclose(result_0.translation, tf0.translation, rtol=1e-6)
        np.testing.assert_allclose(result_0.rotation, tf0.rotation, rtol=1e-6)
        np.testing.assert_allclose(result_1.translation, tf1.translation, rtol=1e-6)
        np.testing.assert_allclose(result_1.rotation, tf1.rotation, rtol=1e-6)

    def test_transform_interpolate_midpoint(self):
        """Transform interpolation at t=0.5 should be midpoint."""
        tf0 = Transform.identity(backend="numpy")
        tf1 = Transform.from_rep(
            np.array([2.0, 0.0, 0.0, 0, 0, 0]),  # Just translation
            from_rep="euler"
        )
        
        result = transform_interpolate(tf0, tf1, 0.5)
        
        # Translation should be midpoint
        np.testing.assert_allclose(result.translation, np.array([1.0, 0.0, 0.0]), rtol=1e-6)

    def test_transform_sequence_interpolate(self):
        """Sequence interpolation should work correctly."""
        # Create keyframes
        tf0 = Transform.identity(backend="numpy")
        tf1 = Transform.from_rep(np.array([1, 0, 0, 0, 0, 0]), from_rep="euler")
        tf2 = Transform.from_rep(np.array([2, 0, 0, 0, 0, 0]), from_rep="euler")
        
        transforms = [tf0, tf1, tf2]
        times = np.array([0.0, 1.0, 2.0])
        query_times = np.array([0.5, 1.5])
        
        result = transform_sequence_interpolate(transforms, times, query_times)
        
        assert result.translation.shape == (2, 3)
        np.testing.assert_allclose(result.translation[0], [0.5, 0, 0], rtol=1e-6)
        np.testing.assert_allclose(result.translation[1], [1.5, 0, 0], rtol=1e-6)


# =============================================================================
# Test: Batch Operations
# =============================================================================


class TestBatchOperations:
    """Test that functions handle arbitrary batch dimensions."""

    def test_quaternion_to_matrix_batched(self):
        """Quaternion to matrix should work with various batch shapes."""
        for shape in [(4,), (5, 4), (2, 3, 4), (2, 3, 4, 4)]:
            quat = np.random.randn(*shape)
            quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
            
            matrix = quaternion_to_matrix(quat)
            
            expected_shape = shape[:-1] + (3, 3)
            assert matrix.shape == expected_shape, f"Shape mismatch for input shape {shape}"

    def test_rotation_6d_batched(self):
        """6D rotation functions should work with various batch shapes."""
        for shape in [(6,), (5, 6), (2, 3, 6)]:
            rot_6d = np.random.randn(*shape)
            
            matrix = rotation_6d_to_matrix(rot_6d)
            
            expected_shape = shape[:-1] + (3, 3)
            assert matrix.shape == expected_shape

    def test_euler_batched_torch(self):
        """Euler functions should work with batched PyTorch tensors."""
        euler = torch.randn(5, 10, 3)
        
        matrix = euler_to_matrix(euler, seq="ZYX")
        
        assert matrix.shape == (5, 10, 3, 3)

    def test_transform_batched_operations(self):
        """Transform operations should work with batched data."""
        rot = np.random.randn(5, 3, 3)
        rot = orthogonalize_rotation(rot)
        trans = np.random.randn(5, 3)
        
        tf = Transform(rotation=rot, translation=trans)
        
        assert tf.batch_shape == (5,)
        assert tf.num_transforms == 5
        
        # Test indexing
        tf_0 = tf[0]
        assert tf_0.rotation.shape == (3, 3)
        assert tf_0.translation.shape == (3,)


# =============================================================================
# Test: Transform Class
# =============================================================================


class TestTransformClass:
    """Test Transform class methods."""

    def test_identity(self):
        """Test identity transform creation."""
        tf_np = Transform.identity(backend="numpy")
        tf_torch = Transform.identity(backend="torch")
        
        np.testing.assert_allclose(tf_np.rotation, np.eye(3))
        np.testing.assert_allclose(tf_np.translation, np.zeros(3))
        
        assert torch.allclose(tf_torch.rotation, torch.eye(3))
        assert torch.allclose(tf_torch.translation, torch.zeros(3))

    def test_from_matrix(self):
        """Test creating Transform from 4x4 matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = [1, 2, 3]
        
        tf = Transform.from_matrix(matrix)
        
        np.testing.assert_allclose(tf.rotation, np.eye(3))
        np.testing.assert_allclose(tf.translation, [1, 2, 3])

    def test_as_matrix(self):
        """Test converting Transform to 4x4 matrix."""
        tf = Transform(
            rotation=np.eye(3),
            translation=np.array([1, 2, 3])
        )
        
        matrix = tf.as_matrix()
        
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3))
        np.testing.assert_allclose(matrix[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(matrix[3, :], [0, 0, 0, 1])

    def test_composition(self):
        """Test transform composition with @ operator."""
        # Translation only
        tf1 = Transform(rotation=np.eye(3), translation=np.array([1, 0, 0]))
        tf2 = Transform(rotation=np.eye(3), translation=np.array([0, 1, 0]))
        
        composed = tf1 @ tf2
        
        np.testing.assert_allclose(composed.translation, [1, 1, 0])

    def test_inverse(self):
        """Test transform inverse."""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler("ZYX", [0.1, 0.2, 0.3]).as_matrix()
        tf = Transform(rotation=rot, translation=np.array([1, 2, 3]))
        
        tf_inv = tf.inverse()
        composed = tf @ tf_inv
        
        np.testing.assert_allclose(composed.rotation, np.eye(3), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(composed.translation, np.zeros(3), rtol=1e-5, atol=1e-6)

    def test_transform_point(self):
        """Test transforming points."""
        # 90 degree rotation around Z + translation
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        tf = Transform(rotation=rot, translation=np.array([1, 0, 0]))
        
        point = np.array([1, 0, 0])
        transformed = tf.transform_point(point)
        
        np.testing.assert_allclose(transformed, [1, 1, 0], rtol=1e-6)

    def test_to_rep_and_from_rep(self):
        """Test conversion to/from representations."""
        tf = Transform.identity(backend="numpy")
        tf.translation = np.array([1, 2, 3])
        
        # To quaternion representation
        quat_rep = tf.to_rep("quat")
        assert quat_rep.shape == (7,)  # 3 translation + 4 quaternion
        
        # Back to Transform
        tf_recovered = Transform.from_rep(quat_rep, from_rep="quat")
        
        np.testing.assert_allclose(tf_recovered.translation, tf.translation)
        np.testing.assert_allclose(tf_recovered.rotation, tf.rotation)

    def test_clone(self):
        """Test cloning transforms."""
        tf = Transform.identity(backend="numpy")
        tf.translation = np.array([1, 2, 3])
        
        tf_clone = tf.clone()
        
        # Should be equal but not same object
        np.testing.assert_allclose(tf_clone.translation, tf.translation)
        assert tf_clone.translation is not tf.translation

    def test_to_device(self):
        """Test moving transform to device (PyTorch)."""
        tf = Transform.identity(backend="torch")
        
        # Move to same device (should not fail)
        tf_moved = tf.to(device="cpu")
        
        assert tf_moved.device.type == "cpu"

    def test_requires_grad(self):
        """Test gradient tracking."""
        tf = Transform.identity(backend="torch")
        
        assert tf.requires_grad is False
        
        tf.requires_grad_(True)
        
        assert tf.requires_grad is True
        assert tf.rotation.requires_grad is True
        assert tf.translation.requires_grad is True

    def test_detach(self):
        """Test detaching from computation graph."""
        tf = Transform.identity(backend="torch").requires_grad_(True)
        tf_detached = tf.detach()
        
        assert tf_detached.requires_grad is False

    def test_repr(self):
        """Test string representation."""
        tf = Transform.identity(backend="numpy")
        repr_str = repr(tf)
        
        assert "Transform" in repr_str
        assert "numpy" in repr_str

    def test_getitem(self):
        """Test indexing batched transforms."""
        rot = np.stack([np.eye(3)] * 5)
        trans = np.random.randn(5, 3)
        tf = Transform(rotation=rot, translation=trans)
        
        tf_0 = tf[0]
        assert tf_0.rotation.shape == (3, 3)
        np.testing.assert_allclose(tf_0.translation, trans[0])
        
        tf_slice = tf[1:3]
        assert tf_slice.rotation.shape == (2, 3, 3)


# =============================================================================
# Test: Utilities
# =============================================================================


class TestUtilities:
    """Test utility functions."""

    def test_geodesic_distance_zero(self, identity_matrix_np):
        """Geodesic distance between identical rotations should be zero."""
        dist = geodesic_distance(identity_matrix_np, identity_matrix_np)
        assert abs(dist) < 1e-6

    def test_geodesic_distance_90_degrees(self):
        """Geodesic distance for 90-degree rotation should be pi/2."""
        R1 = np.eye(3)
        R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)  # 90 deg Z
        
        dist = geodesic_distance(R1, R2)
        
        np.testing.assert_allclose(dist, np.pi / 2, rtol=1e-5)

    def test_geodesic_distance_degrees(self):
        """Geodesic distance should work with degrees output."""
        R1 = np.eye(3)
        R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        
        dist_deg = geodesic_distance(R1, R2, degrees=True)
        
        np.testing.assert_allclose(dist_deg, 90.0, rtol=1e-5)

    def test_orthogonalize_rotation(self):
        """Orthogonalization should produce valid rotation matrices."""
        # Start with slightly non-orthogonal matrix
        R = np.eye(3) + 0.1 * np.random.randn(3, 3)
        
        R_ortho = orthogonalize_rotation(R)
        
        # Check orthogonality
        np.testing.assert_allclose(R_ortho @ R_ortho.T, np.eye(3), rtol=1e-5, atol=1e-6)
        
        # Check determinant
        np.testing.assert_allclose(np.linalg.det(R_ortho), 1.0, rtol=1e-5)

    def test_orthogonalize_rotation_batched(self):
        """Orthogonalization should work with batched matrices."""
        R = np.eye(3) + 0.1 * np.random.randn(5, 3, 3)
        
        R_ortho = orthogonalize_rotation(R)
        
        assert R_ortho.shape == (5, 3, 3)
        for i in range(5):
            np.testing.assert_allclose(
                R_ortho[i] @ R_ortho[i].T, np.eye(3), rtol=1e-5, atol=1e-6
            )


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling and validation."""

    def test_quaternion_wrong_shape(self):
        """Should raise error for wrong quaternion shape."""
        with pytest.raises(ValueError, match="must have shape"):
            quaternion_to_matrix(np.array([1, 2, 3]))  # Missing 4th element

    def test_rotation_6d_wrong_shape(self):
        """Should raise error for wrong 6D rotation shape."""
        with pytest.raises(ValueError, match="must have shape"):
            rotation_6d_to_matrix(np.array([1, 2, 3, 4, 5]))  # Only 5 elements

    def test_transform_backend_mismatch(self):
        """Should raise error for mixing backends in composition."""
        tf_np = Transform.identity(backend="numpy")
        tf_torch = Transform.identity(backend="torch")
        
        with pytest.raises(ValueError, match="different backends"):
            tf_np @ tf_torch

    def test_requires_grad_on_numpy(self):
        """Should raise error for requires_grad on NumPy."""
        tf = Transform.identity(backend="numpy")
        
        with pytest.raises(ValueError, match="only supported for PyTorch"):
            tf.requires_grad_(True)

    def test_to_on_numpy(self):
        """Should raise error for to() on NumPy."""
        tf = Transform.identity(backend="numpy")
        
        with pytest.raises(ValueError, match="only supported for PyTorch"):
            tf.to(device="cpu")


# =============================================================================
# Test Quaternion Operations
# =============================================================================


class TestQuaternionOperations:
    """Tests for quaternion_conjugate, quaternion_inverse, quaternion_multiply, quaternion_apply."""
    
    def test_conjugate_numpy(self):
        """Quaternion conjugate should negate xyz components."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        
        q_conj = quaternion_conjugate(q)
        
        np.testing.assert_allclose(q_conj[:3], -q[:3])
        np.testing.assert_allclose(q_conj[3], q[3])
    
    def test_conjugate_torch(self):
        """Quaternion conjugate should negate xyz components (PyTorch)."""
        q = torch.tensor([0.1, 0.2, 0.3, 0.9])
        q = q / torch.norm(q)
        
        q_conj = quaternion_conjugate(q)
        
        torch.testing.assert_close(q_conj[:3], -q[:3])
        torch.testing.assert_close(q_conj[3], q[3])
    
    def test_inverse_unit_quaternion(self):
        """For unit quaternions, inverse equals conjugate."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        
        q_inv = quaternion_inverse(q)
        q_conj = quaternion_conjugate(q)
        
        np.testing.assert_allclose(q_inv, q_conj, atol=1e-6)
    
    def test_multiply_identity(self):
        """Multiplying by identity should return original."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        
        result = quaternion_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-6)
        
        result2 = quaternion_multiply(identity, q)
        np.testing.assert_allclose(result2, q, atol=1e-6)
    
    def test_multiply_inverse(self):
        """q * q^{-1} should equal identity."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q, q_inv)
        
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, identity, atol=1e-6)
    
    def test_multiply_consistency_with_matrix(self):
        """Quaternion multiplication should match matrix multiplication."""
        q1 = np.array([0.1, 0.2, 0.3, 0.9])
        q1 = q1 / np.linalg.norm(q1)
        q2 = np.array([0.4, -0.1, 0.2, 0.8])
        q2 = q2 / np.linalg.norm(q2)
        
        # Via quaternion multiplication
        q_result = quaternion_multiply(q1, q2)
        m_result = quaternion_to_matrix(q_result)
        
        # Via matrix multiplication
        m1 = quaternion_to_matrix(q1)
        m2 = quaternion_to_matrix(q2)
        m_expected = m1 @ m2
        
        np.testing.assert_allclose(m_result, m_expected, atol=1e-6)
    
    def test_multiply_batched(self):
        """Quaternion multiplication should work with batches."""
        q1 = np.random.randn(5, 4)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
        q2 = np.random.randn(5, 4)
        q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
        
        result = quaternion_multiply(q1, q2)
        assert result.shape == (5, 4)
    
    def test_apply_identity(self):
        """Applying identity quaternion should not change vector."""
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        v = np.array([1.0, 2.0, 3.0])
        
        result = quaternion_apply(identity, v)
        np.testing.assert_allclose(result, v, atol=1e-6)
    
    def test_apply_90_degree_z(self):
        """Rotating [1, 0, 0] by 90 degrees around z should give [0, 1, 0]."""
        angle = np.pi / 2
        q = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])  # 90° around z
        v = np.array([1.0, 0.0, 0.0])
        
        result = quaternion_apply(q, v)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)
    
    def test_apply_consistency_with_matrix(self):
        """quaternion_apply should match matrix @ vector."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        v = np.array([1.0, 2.0, 3.0])
        
        # Via quaternion_apply
        result_quat = quaternion_apply(q, v)
        
        # Via matrix
        m = quaternion_to_matrix(q)
        result_matrix = m @ v
        
        np.testing.assert_allclose(result_quat, result_matrix, atol=1e-6)
    
    def test_apply_batched(self):
        """quaternion_apply should work with batches."""
        q = np.random.randn(5, 4)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        v = np.random.randn(5, 3)
        
        result = quaternion_apply(q, v)
        assert result.shape == (5, 3)
    
    def test_multiply_gradient(self):
        """Gradients should flow through quaternion multiplication."""
        q1 = torch.randn(4, requires_grad=True)
        q1_normalized = q1 / torch.norm(q1)
        q2 = torch.randn(4)
        q2 = q2 / torch.norm(q2)
        
        result = quaternion_multiply(q1_normalized, q2)
        loss = result.sum()
        loss.backward()
        
        assert q1.grad is not None
        assert not torch.isnan(q1.grad).any()


# =============================================================================
# Test Transform Additional Methods
# =============================================================================


class TestTransformAdditionalMethods:
    """Tests for from_pos_quat, stack, apply_delta, relative_to."""
    
    def test_from_pos_quat(self):
        """from_pos_quat should correctly construct transform."""
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([0, 0, 0, 1])  # identity rotation
        
        tf = Transform.from_pos_quat(pos, quat)
        
        np.testing.assert_allclose(tf.translation, pos)
        np.testing.assert_allclose(tf.rotation, np.eye(3), atol=1e-6)
    
    def test_from_pos_quat_batched(self):
        """from_pos_quat should work with batches."""
        pos = np.random.randn(5, 3)
        quat = np.random.randn(5, 4)
        quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        
        tf = Transform.from_pos_quat(pos, quat)
        
        assert tf.rotation.shape == (5, 3, 3)
        assert tf.translation.shape == (5, 3)
    
    def test_stack(self):
        """stack should combine transforms into a batch."""
        tf1 = Transform.identity()
        tf2 = Transform.from_pos_quat(np.array([1, 0, 0]), np.array([0, 0, 0, 1]))
        tf3 = Transform.from_pos_quat(np.array([2, 0, 0]), np.array([0, 0, 0, 1]))
        
        batched = Transform.stack([tf1, tf2, tf3])
        
        assert batched.rotation.shape == (3, 3, 3)
        assert batched.translation.shape == (3, 3)
        np.testing.assert_allclose(batched.translation[0], [0, 0, 0])
        np.testing.assert_allclose(batched.translation[1], [1, 0, 0])
        np.testing.assert_allclose(batched.translation[2], [2, 0, 0])
    
    def test_apply_delta_world_frame(self):
        """apply_delta in world frame should compose as delta @ self."""
        tf = Transform.from_pos_quat(np.array([1, 0, 0]), np.array([0, 0, 0, 1]))
        delta = Transform.from_pos_quat(np.array([0.1, 0, 0]), np.array([0, 0, 0, 1]))
        
        result = tf.apply_delta(delta, in_world_frame=True)
        expected = delta @ tf
        
        np.testing.assert_allclose(result.translation, expected.translation)
        np.testing.assert_allclose(result.rotation, expected.rotation)
    
    def test_apply_delta_local_frame(self):
        """apply_delta in local frame should compose as self @ delta."""
        tf = Transform.from_pos_quat(np.array([1, 0, 0]), np.array([0, 0, 0, 1]))
        delta = Transform.from_pos_quat(np.array([0.1, 0, 0]), np.array([0, 0, 0, 1]))
        
        result = tf.apply_delta(delta, in_world_frame=False)
        expected = tf @ delta
        
        np.testing.assert_allclose(result.translation, expected.translation)
        np.testing.assert_allclose(result.rotation, expected.rotation)
    
    def test_relative_to(self):
        """relative_to should compute transform in reference frame."""
        ref = Transform.from_pos_quat(np.array([1, 0, 0]), np.array([0, 0, 0, 1]))
        tf = Transform.from_pos_quat(np.array([2, 1, 0]), np.array([0, 0, 0, 1]))
        
        relative = tf.relative_to(ref)
        
        # ref @ relative should equal tf
        reconstructed = ref @ relative
        np.testing.assert_allclose(reconstructed.translation, tf.translation, atol=1e-6)
        np.testing.assert_allclose(reconstructed.rotation, tf.rotation, atol=1e-6)


# =============================================================================
# Test SE(3) Lie Group Operations
# =============================================================================


class TestSE3Operations:
    """Tests for se3_log and se3_exp."""
    
    def test_exp_log_roundtrip_identity(self):
        """exp(log(identity)) should return identity."""
        tf = Transform.identity()
        
        twist = se3_log(tf)
        tf_recovered = se3_exp(twist)
        
        np.testing.assert_allclose(tf_recovered.rotation, tf.rotation, atol=1e-6)
        np.testing.assert_allclose(tf_recovered.translation, tf.translation, atol=1e-6)
    
    def test_exp_log_roundtrip_pure_translation(self):
        """exp(log(pure translation)) should recover transform."""
        tf = Transform.from_pos_quat(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1]))
        
        twist = se3_log(tf)
        tf_recovered = se3_exp(twist)
        
        np.testing.assert_allclose(tf_recovered.rotation, tf.rotation, atol=1e-6)
        np.testing.assert_allclose(tf_recovered.translation, tf.translation, atol=1e-6)
    
    def test_exp_log_roundtrip_small_rotation(self):
        """exp(log(small rotation)) should recover transform."""
        angle = 0.1  # small angle
        quat = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
        tf = Transform.from_pos_quat(np.array([0.5, 0.5, 0.5]), quat)
        
        twist = se3_log(tf)
        tf_recovered = se3_exp(twist)
        
        np.testing.assert_allclose(tf_recovered.rotation, tf.rotation, atol=1e-5)
        np.testing.assert_allclose(tf_recovered.translation, tf.translation, atol=1e-5)
    
    def test_exp_zero_twist_is_identity(self):
        """exp(0) should be identity transform."""
        twist = np.zeros(6)
        
        tf = se3_exp(twist)
        
        np.testing.assert_allclose(tf.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(tf.translation, np.zeros(3), atol=1e-6)
    
    def test_log_identity_is_zero(self):
        """log(identity) should be zero twist."""
        tf = Transform.identity()
        
        twist = se3_log(tf)
        
        np.testing.assert_allclose(twist, np.zeros(6), atol=1e-6)
    
    def test_se3_torch_gradient(self):
        """Gradients should flow through SE(3) operations."""
        twist = torch.randn(6, requires_grad=True)
        
        tf = se3_exp(twist)
        loss = tf.rotation.sum() + tf.translation.sum()
        loss.backward()
        
        assert twist.grad is not None
        assert not torch.isnan(twist.grad).any()

# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
