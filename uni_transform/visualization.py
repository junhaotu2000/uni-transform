"""
Visualization utilities using Rerun.

Provides visualization for:
- Coordinate frames (Transform)
- Transform graphs (TransformManager)  
- Trajectories (sequence of Transforms)
- Point clouds in different frames

Requires: pip install rerun-sdk

Reference: https://github.com/rerun-io/rerun
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from .transform import Transform
    
    # TransformManager is optional
    try:
        from .transform_manager import TransformManager
    except ImportError:
        TransformManager = None  # type: ignore

# Lazy import rerun to avoid hard dependency
_rr = None


def _get_rerun():
    """Lazy import rerun."""
    global _rr
    if _rr is None:
        try:
            import rerun as rr
            _rr = rr
        except ImportError:
            raise ImportError(
                "rerun-sdk is required for visualization. "
                "Install with: pip install rerun-sdk"
            )
    return _rr


# ==================== Core Visualization ====================


def init(
    app_name: str = "uni_transform",
    *,
    spawn: bool = False,
    save_path: Optional[str] = None,
    connect_addr: Optional[str] = None,
    recording_id: Optional[str] = None,
) -> None:
    """
    Initialize Rerun visualization.
    
    Args:
        app_name: Name of the application/recording.
        spawn: If True, spawn the Rerun viewer (requires GUI environment).
        save_path: If provided, save recording to this .rrd file.
        connect_addr: If provided, connect to remote viewer at this address.
        recording_id: Optional recording ID for multiple sessions.
    
    Modes (mutually exclusive, checked in order):
        1. save_path: Save to file (headless, for later viewing)
        2. connect_addr: Connect to remote viewer
        3. spawn=True: Launch local viewer (requires GUI)
        4. Default: Memory recording (call save() later)
    
    Example:
        >>> # Headless server - save to file
        >>> init("robot_viz", save_path="recording.rrd")
        >>> 
        >>> # Connect to remote viewer
        >>> init("robot_viz", connect_addr="192.168.1.100:9876")
        >>> 
        >>> # Local GUI environment
        >>> init("robot_viz", spawn=True)
    """
    rr = _get_rerun()
    rr.init(app_name, recording_id=recording_id)
    
    if save_path:
        rr.save(save_path)
    elif connect_addr:
        rr.connect_grpc(f"rerun+http://{connect_addr}/proxy")
    elif spawn:
        try:
            rr.spawn()
        except RuntimeError as e:
            if "Failed to find Rerun Viewer" in str(e):
                raise RuntimeError(
                    f"Cannot spawn Rerun viewer (headless environment?). "
                    f"Use save_path='recording.rrd' to save for later viewing, "
                    f"or connect_addr='host:port' to connect to a remote viewer.\n"
                    f"Original error: {e}"
                ) from e
            raise


def connect(addr: str = "127.0.0.1:9876") -> None:
    """Connect to a running Rerun viewer via gRPC."""
    rr = _get_rerun()
    rr.connect_grpc(f"rerun+http://{addr}/proxy")


def save(path: str) -> None:
    """Save recording to a .rrd file."""
    rr = _get_rerun()
    rr.save(path)


# ==================== Transform Visualization ====================


def log_transform(
    entity_path: str,
    transform: "Transform",
    *,
    axis_length: float = 0.1,
    axis_radius: float = 0.005,
    show_label: bool = True,
    static: bool = False,
) -> None:
    """
    Log a Transform as a coordinate frame.
    
    Visualizes the transform as 3 arrows (X=red, Y=green, Z=blue).
    
    Args:
        entity_path: Rerun entity path (e.g., "world/robot/base").
        transform: Transform to visualize.
        axis_length: Length of coordinate axes.
        axis_radius: Radius of axis arrows.
        show_label: Whether to show the entity name as a label.
        static: If True, the transform persists across all time.
    
    Example:
        >>> log_transform("robot/base", base_transform)
        >>> log_transform("robot/tool", tool_transform, axis_length=0.05)
    """
    rr = _get_rerun()
    
    # Convert to numpy if needed
    rotation = transform.rotation
    translation = transform.translation
    if hasattr(rotation, 'numpy'):
        rotation = rotation.detach().cpu().numpy()
    if hasattr(translation, 'numpy'):
        translation = translation.detach().cpu().numpy()
    
    # Ensure proper shape
    rotation = np.asarray(rotation).reshape(3, 3)
    translation = np.asarray(translation).reshape(3)
    
    # Build 4x4 matrix for Rerun
    mat4x4 = np.eye(4)
    mat4x4[:3, :3] = rotation
    mat4x4[:3, 3] = translation
    
    # Log the transform
    rr.log(
        entity_path,
        rr.Transform3D(mat3x3=rotation, translation=translation),
        static=static,
    )
    
    # Draw coordinate axes as arrows
    origin = np.zeros((3, 3))
    directions = rotation.T * axis_length  # Column vectors are axes
    colors = np.array([
        [255, 0, 0, 255],    # X = Red
        [0, 255, 0, 255],    # Y = Green
        [0, 0, 255, 255],    # Z = Blue
    ], dtype=np.uint8)
    
    rr.log(
        f"{entity_path}/axes",
        rr.Arrows3D(
            origins=origin,
            vectors=directions,
            colors=colors,
            radii=axis_radius,
        ),
        static=static,
    )
    
    # Optional label
    if show_label:
        label = entity_path.split("/")[-1]
        rr.log(
            f"{entity_path}/label",
            rr.TextLog(label),
            static=static,
        )


def log_trajectory(
    entity_path: str,
    transforms: Sequence["Transform"],
    *,
    color: Optional[tuple] = None,
    radius: float = 0.002,
    show_frames: bool = False,
    frame_interval: int = 10,
    axis_length: float = 0.05,
) -> None:
    """
    Log a trajectory as a 3D line strip.
    
    Args:
        entity_path: Rerun entity path.
        transforms: Sequence of transforms representing the trajectory.
        color: RGBA color tuple (0-255). Default: orange.
        radius: Line radius.
        show_frames: Whether to show coordinate frames along the trajectory.
        frame_interval: Show a frame every N transforms.
        axis_length: Length of frame axes if show_frames=True.
    
    Example:
        >>> trajectory = [t1, t2, t3, t4, t5]
        >>> log_trajectory("robot/path", trajectory, color=(255, 128, 0, 255))
    """
    rr = _get_rerun()
    
    if not transforms:
        return
    
    # Extract positions
    positions = []
    for t in transforms:
        trans = t.translation
        if hasattr(trans, 'numpy'):
            trans = trans.detach().cpu().numpy()
        positions.append(np.asarray(trans).reshape(3))
    
    positions = np.array(positions)
    
    if color is None:
        color = (255, 128, 0, 255)  # Orange
    
    # Log trajectory line
    rr.log(
        f"{entity_path}/path",
        rr.LineStrips3D(
            [positions],
            colors=[color],
            radii=radius,
        ),
    )
    
    # Log start and end points
    rr.log(
        f"{entity_path}/start",
        rr.Points3D(
            positions=positions[:1],
            colors=[(0, 255, 0, 255)],  # Green
            radii=radius * 3,
        ),
    )
    rr.log(
        f"{entity_path}/end",
        rr.Points3D(
            positions=positions[-1:],
            colors=[(255, 0, 0, 255)],  # Red
            radii=radius * 3,
        ),
    )
    
    # Optionally show frames along trajectory
    if show_frames:
        for i, t in enumerate(transforms):
            if i % frame_interval == 0:
                log_transform(
                    f"{entity_path}/frames/{i}",
                    t,
                    axis_length=axis_length,
                    show_label=False,
                )


def log_trajectory_animated(
    entity_path: str,
    transforms: Sequence["Transform"],
    *,
    timeline: str = "frame",
    start_time: int = 0,
    color: Optional[tuple] = None,
    axis_length: float = 0.1,
) -> None:
    """
    Log a trajectory with time animation.
    
    Each transform is logged at a different time step, allowing
    playback in the Rerun viewer.
    
    Args:
        entity_path: Rerun entity path.
        transforms: Sequence of transforms.
        timeline: Name of the timeline.
        start_time: Starting time/frame number.
        color: RGBA color for the path trail.
        axis_length: Length of coordinate axes.
    
    Example:
        >>> log_trajectory_animated("robot/pose", trajectory)
        >>> # Then use the timeline slider in Rerun viewer
    """
    rr = _get_rerun()
    
    if color is None:
        color = (100, 100, 255, 128)  # Semi-transparent blue
    
    positions = []
    for i, t in enumerate(transforms):
        rr.set_time(timeline, sequence=start_time + i)
        
        # Log current pose
        log_transform(entity_path, t, axis_length=axis_length, show_label=False)
        
        # Accumulate path
        trans = t.translation
        if hasattr(trans, 'numpy'):
            trans = trans.detach().cpu().numpy()
        positions.append(np.asarray(trans).reshape(3))
        
        # Log trail
        if len(positions) > 1:
            rr.log(
                f"{entity_path}/trail",
                rr.LineStrips3D(
                    [np.array(positions)],
                    colors=[color],
                    radii=0.002,
                ),
            )


# ==================== TransformManager Visualization ====================


def log_transform_manager(
    entity_path: str,
    manager: "TransformManager",
    *,
    root_frame: Optional[str] = None,
    axis_length: float = 0.1,
    edge_color: Optional[tuple] = None,
    edge_radius: float = 0.003,
    static: bool = True,
) -> None:
    """
    Log a TransformManager as a tree of coordinate frames.
    
    Args:
        entity_path: Base entity path.
        manager: TransformManager to visualize.
        root_frame: Root frame to start from. If None, auto-detect.
        axis_length: Length of coordinate axes.
        edge_color: Color for edges between frames.
        edge_radius: Radius of edge lines.
        static: If True, persists across all time.
    
    Example:
        >>> tm = TransformManager()
        >>> tm.add("base", "shoulder", t1).add("shoulder", "elbow", t2)
        >>> log_transform_manager("robot", tm, root_frame="base")
    """
    rr = _get_rerun()
    
    if edge_color is None:
        edge_color = (150, 150, 150, 200)  # Gray
    
    if not manager.frames:
        return
    
    # Find root (frame with most connections, or specified)
    if root_frame is None:
        root_frame = max(
            manager.frames,
            key=lambda f: len(manager.neighbors(f))
        )
    
    # BFS to build tree and log frames
    from collections import deque
    
    visited = set()
    queue = deque([(root_frame, np.eye(4))])  # (frame, world_transform)
    frame_positions: Dict[str, np.ndarray] = {}
    edges: List[tuple] = []
    
    # Import Transform for identity
    from .transform import Transform
    
    while queue:
        frame, world_mat = queue.popleft()
        if frame in visited:
            continue
        visited.add(frame)
        
        # World position
        world_pos = world_mat[:3, 3]
        frame_positions[frame] = world_pos
        
        # Create transform from world matrix
        world_transform = Transform(
            rotation=world_mat[:3, :3],
            translation=world_pos,
        )
        
        # Log this frame
        log_transform(
            f"{entity_path}/{frame}",
            world_transform,
            axis_length=axis_length,
            show_label=True,
            static=static,
        )
        
        # Process children
        for neighbor in manager.neighbors(frame):
            if neighbor not in visited:
                # Get relative transform
                try:
                    rel_transform = manager.get(frame, neighbor)
                    rel_rot = rel_transform.rotation
                    rel_trans = rel_transform.translation
                    if hasattr(rel_rot, 'numpy'):
                        rel_rot = rel_rot.detach().cpu().numpy()
                    if hasattr(rel_trans, 'numpy'):
                        rel_trans = rel_trans.detach().cpu().numpy()
                    
                    # Compute world transform for child
                    rel_mat = np.eye(4)
                    rel_mat[:3, :3] = np.asarray(rel_rot).reshape(3, 3)
                    rel_mat[:3, 3] = np.asarray(rel_trans).reshape(3)
                    child_world_mat = world_mat @ rel_mat
                    
                    queue.append((neighbor, child_world_mat))
                    edges.append((frame, neighbor))
                except KeyError:
                    continue
    
    # Log edges between frames
    if edges:
        edge_lines = []
        for parent, child in edges:
            if parent in frame_positions and child in frame_positions:
                edge_lines.append([
                    frame_positions[parent],
                    frame_positions[child],
                ])
        
        if edge_lines:
            rr.log(
                f"{entity_path}/_edges",
                rr.LineStrips3D(
                    edge_lines,
                    colors=[edge_color] * len(edge_lines),
                    radii=edge_radius,
                ),
                static=static,
            )


# ==================== Point Cloud Visualization ====================


def log_points(
    entity_path: str,
    points: np.ndarray,
    *,
    colors: Optional[np.ndarray] = None,
    radii: Optional[Union[float, np.ndarray]] = None,
    transform: Optional["Transform"] = None,
) -> None:
    """
    Log 3D points, optionally transformed.
    
    Args:
        entity_path: Rerun entity path.
        points: Points array of shape (N, 3).
        colors: Optional colors array of shape (N, 3) or (N, 4).
        radii: Point radii (scalar or per-point array).
        transform: Optional transform to apply to points.
    
    Example:
        >>> points = np.random.randn(100, 3)
        >>> log_points("scene/cloud", points)
        >>> log_points("robot/cloud", points, transform=robot_pose)
    """
    rr = _get_rerun()
    
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    
    # Apply transform if provided
    if transform is not None:
        points = transform.transform_point(points)
        if hasattr(points, 'numpy'):
            points = points.detach().cpu().numpy()
    
    kwargs = {}
    if colors is not None:
        kwargs['colors'] = colors
    if radii is not None:
        kwargs['radii'] = radii
    
    rr.log(entity_path, rr.Points3D(positions=points, **kwargs))


# ==================== Utility ====================


def log_text(entity_path: str, text: str) -> None:
    """Log text annotation."""
    rr = _get_rerun()
    rr.log(entity_path, rr.TextLog(text))


def set_time(timeline: str, value: int) -> None:
    """Set the current time on a timeline."""
    rr = _get_rerun()
    rr.set_time(timeline, sequence=value)


def log_scalar(entity_path: str, value: float) -> None:
    """Log a scalar value for time series plotting."""
    rr = _get_rerun()
    rr.log(entity_path, rr.Scalar(value))


# ==================== High-Level API ====================


class RerunVisualizer:
    """
    High-level visualizer for uni_transform objects.
    
    Example:
        >>> # Headless - save to file
        >>> viz = RerunVisualizer("my_app", save_path="output.rrd")
        >>> viz.show_transform("base", transform)
        >>> 
        >>> # With GUI
        >>> viz = RerunVisualizer("my_app", spawn=True)
    """
    
    def __init__(
        self,
        app_name: str = "uni_transform",
        *,
        spawn: bool = False,
        save_path: Optional[str] = None,
        connect_addr: Optional[str] = None,
    ):
        """
        Initialize the visualizer.
        
        Args:
            app_name: Application name for Rerun.
            spawn: Whether to spawn the Rerun viewer (requires GUI).
            save_path: Save recording to this file (headless mode).
            connect_addr: Connect to remote viewer at this address.
        """
        init(app_name, spawn=spawn, save_path=save_path, connect_addr=connect_addr)
        self._frame_count = 0
    
    def show_transform(
        self,
        name: str,
        transform: "Transform",
        **kwargs,
    ) -> "RerunVisualizer":
        """Show a transform as a coordinate frame."""
        log_transform(name, transform, **kwargs)
        return self
    
    def show_trajectory(
        self,
        name: str,
        transforms: Sequence["Transform"],
        **kwargs,
    ) -> "RerunVisualizer":
        """Show a trajectory."""
        log_trajectory(name, transforms, **kwargs)
        return self
    
    def show_manager(
        self,
        name: str,
        manager: "TransformManager",
        **kwargs,
    ) -> "RerunVisualizer":
        """Show a TransformManager graph."""
        log_transform_manager(name, manager, **kwargs)
        return self
    
    def show_points(
        self,
        name: str,
        points: np.ndarray,
        **kwargs,
    ) -> "RerunVisualizer":
        """Show 3D points."""
        log_points(name, points, **kwargs)
        return self
    
    def animate(
        self,
        name: str,
        transforms: Sequence["Transform"],
        **kwargs,
    ) -> "RerunVisualizer":
        """Animate a trajectory over time."""
        log_trajectory_animated(name, transforms, **kwargs)
        return self
    
    def next_frame(self) -> "RerunVisualizer":
        """Advance to the next frame."""
        self._frame_count += 1
        set_time("frame", self._frame_count)
        return self
    
    def save(self, path: str) -> "RerunVisualizer":
        """Save recording to file."""
        save(path)
        return self

