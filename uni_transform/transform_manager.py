"""
TransformManager: Lightweight coordinate frame graph management.

Features:
- Manage transforms between multiple coordinate frames
- Automatic shortest-path chained transform computation
- Support for both NumPy and PyTorch backends

Example:
    >>> tm = TransformManager()
    >>> tm.add("base", "shoulder", t1)
    >>> tm.add("shoulder", "elbow", t2)
    >>> tm.add("elbow", "wrist", t3)
    >>> 
    >>> # Automatically computes: base → shoulder → elbow → wrist
    >>> base_to_wrist = tm.get("base", "wrist")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .transform import Transform


@dataclass
class TransformManager:
    """
    Manages transforms between coordinate frames as a graph.
    
    Automatically computes chained transforms via shortest path.
    Thread-safe for reads, not for concurrent writes.
    
    Attributes:
        strict_mode: If True, raise error when overwriting existing transforms.
                     If False, silently overwrite. Default: False.
    """
    
    strict_mode: bool = False
    _transforms: Dict[Tuple[str, str], Transform] = field(
        default_factory=dict, repr=False
    )
    _adjacency: Dict[str, Set[str]] = field(default_factory=dict, repr=False)
    _path_cache: Dict[Tuple[str, str], List[str]] = field(
        default_factory=dict, repr=False
    )
    
    # ==================== Core API ====================
    
    def add(
        self,
        from_frame: str,
        to_frame: str,
        transform: Transform,
        *,
        bidirectional: bool = True,
    ) -> "TransformManager":
        """
        Add a transform between two frames.
        
        Args:
            from_frame: Source coordinate frame name.
            to_frame: Target coordinate frame name.
            transform: Transform from source to target.
            bidirectional: If True, also add inverse transform (default: True).
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If strict_mode is True and transform already exists.
        
        Example:
            >>> tm = TransformManager()
            >>> tm.add("base", "camera", camera_pose)
            >>> tm.add("camera", "object", object_pose)
        """
        if from_frame == to_frame:
            raise ValueError(f"Cannot add transform from '{from_frame}' to itself")
        
        key = (from_frame, to_frame)
        if self.strict_mode and key in self._transforms:
            raise ValueError(
                f"Transform '{from_frame}' → '{to_frame}' already exists. "
                "Set strict_mode=False to allow overwriting."
            )
        
        # Store transform
        self._transforms[key] = transform
        
        # Update adjacency graph
        if from_frame not in self._adjacency:
            self._adjacency[from_frame] = set()
        if to_frame not in self._adjacency:
            self._adjacency[to_frame] = set()
        self._adjacency[from_frame].add(to_frame)
        
        # Add inverse if bidirectional
        if bidirectional:
            inv_key = (to_frame, from_frame)
            self._transforms[inv_key] = transform.inverse()
            self._adjacency[to_frame].add(from_frame)
        
        # Invalidate path cache (could be smarter but keep it simple)
        self._path_cache.clear()
        
        return self
    
    def get(self, from_frame: str, to_frame: str) -> Transform:
        """
        Get transform between two frames (automatically chains if needed).
        
        Args:
            from_frame: Source coordinate frame.
            to_frame: Target coordinate frame.
        
        Returns:
            Transform from source to target frame.
        
        Raises:
            KeyError: If no path exists between frames.
        
        Example:
            >>> # If path exists: base → shoulder → elbow → wrist
            >>> t = tm.get("base", "wrist")  # Automatically chains all
        """
        if from_frame == to_frame:
            # Identity transform - get backend from any existing transform
            if self._transforms:
                sample = next(iter(self._transforms.values()))
                return Transform.identity(
                    backend=sample.backend, 
                    translation_unit=sample.translation_unit
                )
            return Transform.identity()
        
        # Direct lookup first (O(1) common case)
        key = (from_frame, to_frame)
        if key in self._transforms:
            return self._transforms[key]
        
        # Find path and chain transforms
        path = self._find_path(from_frame, to_frame)
        return self._chain_path(path)
    
    def has_path(self, from_frame: str, to_frame: str) -> bool:
        """Check if a path exists between two frames."""
        if from_frame == to_frame:
            return True
        if from_frame not in self._adjacency or to_frame not in self._adjacency:
            return False
        try:
            self._find_path(from_frame, to_frame)
            return True
        except KeyError:
            return False
    
    def remove(self, from_frame: str, to_frame: str) -> "TransformManager":
        """
        Remove a transform between two frames.
        
        Args:
            from_frame: Source frame.
            to_frame: Target frame.
        
        Returns:
            self (for method chaining)
        """
        key = (from_frame, to_frame)
        inv_key = (to_frame, from_frame)
        
        if key in self._transforms:
            del self._transforms[key]
            self._adjacency[from_frame].discard(to_frame)
        
        if inv_key in self._transforms:
            del self._transforms[inv_key]
            self._adjacency[to_frame].discard(from_frame)
        
        self._path_cache.clear()
        return self
    
    # ==================== Query API ====================
    
    @property
    def frames(self) -> Set[str]:
        """Get all registered frame names."""
        return set(self._adjacency.keys())
    
    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Get all direct transform edges (without inverses)."""
        seen = set()
        result = []
        for (a, b) in self._transforms.keys():
            edge = tuple(sorted([a, b]))
            if edge not in seen:
                seen.add(edge)
                result.append((a, b))
        return result
    
    def neighbors(self, frame: str) -> Set[str]:
        """Get frames directly connected to the given frame."""
        return self._adjacency.get(frame, set()).copy()
    
    def path(self, from_frame: str, to_frame: str) -> List[str]:
        """
        Get the frame sequence for a path (for debugging/visualization).
        
        Returns:
            List of frame names from source to target.
        
        Example:
            >>> tm.path("base", "wrist")
            ['base', 'shoulder', 'elbow', 'wrist']
        """
        if from_frame == to_frame:
            return [from_frame]
        return self._find_path(from_frame, to_frame)
    
    def __len__(self) -> int:
        """Number of frames in the graph."""
        return len(self._adjacency)
    
    def __contains__(self, frame: str) -> bool:
        """Check if a frame exists in the graph."""
        return frame in self._adjacency
    
    def __repr__(self) -> str:
        n_frames = len(self._adjacency)
        if n_frames == 0:
            return "TransformManager(empty)"
        
        # Build edge strings: "A ↔ B"
        seen = set()
        edge_strs = []
        for (a, b) in self._transforms.keys():
            edge = tuple(sorted([a, b]))
            if edge not in seen:
                seen.add(edge)
                edge_strs.append(f"{a} ↔ {b}")
        
        # Truncate if too many edges
        if len(edge_strs) > 6:
            edges_repr = ", ".join(edge_strs[:5]) + f", ... (+{len(edge_strs) - 5} more)"
        else:
            edges_repr = ", ".join(edge_strs)
        
        return f"TransformManager({edges_repr})"
    
    def __str__(self) -> str:
        """Tree-like visualization of the transform graph."""
        n_frames = len(self._adjacency)
        if n_frames == 0:
            return "TransformManager(empty)"
        
        lines: List[str] = []
        visited: Set[str] = set()
        
        def _build_tree(frame: str, prefix: str, is_last: bool, is_root: bool) -> None:
            """Recursively build tree representation."""
            visited.add(frame)
            
            # Current node
            if is_root:
                lines.append(frame)
            else:
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{frame}")
            
            # Get unvisited children
            children = sorted([n for n in self._adjacency.get(frame, []) if n not in visited])
            
            # Child prefix
            if is_root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")
            
            # Recurse to children
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                _build_tree(child, child_prefix, is_last_child, False)
        
        # Find roots (frames with most connections, or alphabetically first)
        # Start from the frame with most connections as root
        root_candidates = sorted(
            self._adjacency.keys(),
            key=lambda f: (-len(self._adjacency[f]), f)
        )
        
        # Build trees for each connected component
        for root in root_candidates:
            if root not in visited:
                if lines:  # Add separator between components
                    lines.append("")
                _build_tree(root, "", True, True)
        
        return "\n".join(lines)
    
    # ==================== Internal ====================
    
    def _find_path(self, from_frame: str, to_frame: str) -> List[str]:
        """BFS to find shortest path between frames."""
        cache_key = (from_frame, to_frame)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        if from_frame not in self._adjacency:
            raise KeyError(f"Frame '{from_frame}' not found in graph")
        if to_frame not in self._adjacency:
            raise KeyError(f"Frame '{to_frame}' not found in graph")
        
        # BFS
        queue = deque([(from_frame, [from_frame])])
        visited = {from_frame}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self._adjacency.get(current, []):
                if neighbor == to_frame:
                    result = path + [neighbor]
                    self._path_cache[cache_key] = result
                    return result
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        raise KeyError(f"No path from '{from_frame}' to '{to_frame}'")
    
    def _chain_path(self, path: List[str]) -> Transform:
        """Chain transforms along a path."""
        if len(path) < 2:
            raise ValueError("Path must have at least 2 frames")
        
        result = self._transforms[(path[0], path[1])]
        for i in range(1, len(path) - 1):
            result = result @ self._transforms[(path[i], path[i + 1])]
        
        return result
    
    # ==================== Convenience ====================
    
    def chain(self, *frames: str) -> Transform:
        """
        Explicitly chain transforms through specified frames.
        
        Useful when you want a specific path instead of shortest path.
        
        Args:
            *frames: Frame names in order (at least 2).
        
        Returns:
            Chained transform from first to last frame.
        
        Example:
            >>> # Force path through specific frames
            >>> t = tm.chain("base", "shoulder", "tool")
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to chain")
        
        result = self.get(frames[0], frames[1])
        for i in range(1, len(frames) - 1):
            result = result @ self.get(frames[i], frames[i + 1])
        
        return result
    
    def transform_point(
        self,
        point,
        from_frame: str,
        to_frame: str,
    ):
        """
        Transform a point from one frame to another.
        
        Args:
            point: Point(s) to transform, shape (..., 3)
            from_frame: Source frame.
            to_frame: Target frame.
        
        Returns:
            Transformed point(s) in target frame.
        """
        t = self.get(from_frame, to_frame)
        return t.transform_point(point)
    
    def clear(self) -> "TransformManager":
        """Remove all transforms and frames."""
        self._transforms.clear()
        self._adjacency.clear()
        self._path_cache.clear()
        return self

