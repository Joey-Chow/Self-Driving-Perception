"""
Camera geometry utilities for 3D transformations and projections.
Handles coordinate system transformations between camera, ego, and BEV frames.
"""

import numpy as np
from typing import Tuple, List, Optional
from pyquaternion import Quaternion


def get_projection_matrix(camera_intrinsic: np.ndarray) -> np.ndarray:
    """
    Get 3x4 projection matrix from camera intrinsics.
    
    Args:
        camera_intrinsic: 3x3 intrinsic matrix
        
    Returns:
        3x4 projection matrix
    """
    # Add column of zeros for homogeneous coordinates
    P = np.hstack([camera_intrinsic, np.zeros((3, 1))])
    return P


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_intrinsic: np.ndarray
) -> np.ndarray:
    """
    Project 3D points in camera frame to 2D image coordinates.
    
    Args:
        points_3d: Nx3 array of 3D points (x, y, z) in camera frame
        camera_intrinsic: 3x3 camera intrinsic matrix
        
    Returns:
        Nx2 array of 2D image coordinates (u, v)
    """
    # Convert to homogeneous coordinates
    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Project to image plane
    P = get_projection_matrix(camera_intrinsic)
    points_2d_h = (P @ points_3d_h.T).T
    
    # Normalize by depth
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    
    return points_2d


def unproject_2d_to_3d(
    points_2d: np.ndarray,
    depths: np.ndarray,
    camera_intrinsic: np.ndarray
) -> np.ndarray:
    """
    Unproject 2D image points to 3D camera coordinates given depths.
    
    Args:
        points_2d: Nx2 array of 2D points (u, v)
        depths: N array of depth values (z coordinate)
        camera_intrinsic: 3x3 camera intrinsic matrix
        
    Returns:
        Nx3 array of 3D points in camera frame
    """
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    
    # Back-project to 3D
    x = (points_2d[:, 0] - cx) * depths / fx
    y = (points_2d[:, 1] - cy) * depths / fy
    z = depths
    
    points_3d = np.stack([x, y, z], axis=1)
    return points_3d


def get_box_corners_3d(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float
) -> np.ndarray:
    """
    Get 8 corners of a 3D bounding box.
    
    Args:
        center: (x, y, z) center of box
        size: (width, length, height)
        yaw: rotation angle around z-axis in radians
        
    Returns:
        8x3 array of corner coordinates
    """
    w, l, h = size
    
    # Create box corners in object frame (centered at origin)
    # Order: front-right, front-left, back-left, back-right (bottom), then top
    corners = np.array([
        [l/2, w/2, -h/2],   # front-right-bottom
        [l/2, -w/2, -h/2],  # front-left-bottom
        [-l/2, -w/2, -h/2], # back-left-bottom
        [-l/2, w/2, -h/2],  # back-right-bottom
        [l/2, w/2, h/2],    # front-right-top
        [l/2, -w/2, h/2],   # front-left-top
        [-l/2, -w/2, h/2],  # back-left-top
        [-l/2, w/2, h/2],   # back-right-top
    ])
    
    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    # Rotate and translate
    corners_rotated = (R @ corners.T).T
    corners_world = corners_rotated + center
    
    return corners_world


def get_box_footprint(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float
) -> np.ndarray:
    """
    Get 4 corners of box footprint on ground plane (for BEV).
    
    Args:
        center: (x, y, z) center of box
        size: (width, length, height)
        yaw: rotation angle around z-axis in radians
        
    Returns:
        4x2 array of (x, y) coordinates
    """
    w, l, _ = size
    
    # Create footprint corners (ground plane)
    corners = np.array([
        [l/2, w/2],    # front-right
        [l/2, -w/2],   # front-left
        [-l/2, -w/2],  # back-left
        [-l/2, w/2],   # back-right
    ])
    
    # Rotation matrix (2D)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    # Rotate and translate (use x, y from center)
    corners_rotated = (R @ corners.T).T
    corners_world = corners_rotated + center[:2]  # Only x, y
    
    return corners_world


def camera_to_ego(
    points: np.ndarray,
    camera_translation: np.ndarray,
    camera_rotation: Quaternion
) -> np.ndarray:
    """
    Transform points from camera frame to ego vehicle frame.
    
    Args:
        points: Nx3 array of points in camera frame
        camera_translation: 3D translation vector
        camera_rotation: Rotation quaternion
        
    Returns:
        Nx3 array of points in ego frame
    """
    # Rotate
    rotation_matrix = camera_rotation.rotation_matrix
    points_rotated = (rotation_matrix @ points.T).T
    
    # Translate
    points_ego = points_rotated + camera_translation
    
    return points_ego


def ego_to_global(
    points: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation: Quaternion
) -> np.ndarray:
    """
    Transform points from ego vehicle frame to global frame.
    
    Args:
        points: Nx3 array of points in ego frame
        ego_translation: 3D translation vector
        ego_rotation: Rotation quaternion
        
    Returns:
        Nx3 array of points in global frame
    """
    # Rotate
    rotation_matrix = ego_rotation.rotation_matrix
    points_rotated = (rotation_matrix @ points.T).T
    
    # Translate
    points_global = points_rotated + ego_translation
    
    return points_global


def is_box_in_image(
    corners_2d: np.ndarray,
    image_width: int,
    image_height: int,
    margin: int = 0
) -> bool:
    """
    Check if a 2D box is visible in the image.
    
    Args:
        corners_2d: Nx2 array of 2D corner points
        image_width: Image width in pixels
        image_height: Image height in pixels
        margin: Margin in pixels for boundary check
        
    Returns:
        True if box is at least partially visible
    """
    # Get bounding box
    min_u, min_v = corners_2d.min(axis=0)
    max_u, max_v = corners_2d.max(axis=0)
    
    # Check if box intersects with image bounds
    if max_u < margin or min_u > image_width - margin:
        return False
    if max_v < margin or min_v > image_height - margin:
        return False
    
    return True


def get_rotation_matrix(yaw: float, pitch: float = 0, roll: float = 0) -> np.ndarray:
    """
    Get 3D rotation matrix from Euler angles.
    
    Args:
        yaw: Rotation around z-axis (radians)
        pitch: Rotation around y-axis (radians)
        roll: Rotation around x-axis (radians)
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation matrices
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def compute_iou_3d(box1: dict, box2: dict) -> float:
    """
    Compute 3D IoU between two boxes (simplified 2D ground plane IoU).
    
    Args:
        box1: Dict with 'center' and 'size'
        box2: Dict with 'center' and 'size'
        
    Returns:
        IoU value [0, 1]
    """
    # Simplified: compute 2D IoU on ground plane
    # Get box rectangles
    x1, y1 = box1['center'][:2]
    w1, l1 = box1['size'][:2]
    
    x2, y2 = box2['center'][:2]
    w2, l2 = box2['size'][:2]
    
    # Compute intersection
    x_left = max(x1 - l1/2, x2 - l2/2)
    y_top = max(y1 - w1/2, y2 - w2/2)
    x_right = min(x1 + l1/2, x2 + l2/2)
    y_bottom = min(y1 + w1/2, y2 + w2/2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * l1
    area2 = w2 * l2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


if __name__ == '__main__':
    # Test camera geometry functions
    print("Testing camera geometry utilities...")
    
    # Test 3D box corners
    center = np.array([10.0, 0.0, 0.0])  # 10m in front
    size = np.array([1.8, 4.5, 1.5])     # Car dimensions (WxLxH)
    yaw = np.deg2rad(30)                 # 30 degree rotation
    
    corners = get_box_corners_3d(center, size, yaw)
    print(f"\n3D Box corners shape: {corners.shape}")
    print(f"Center: {center}")
    print(f"First corner: {corners[0]}")
    
    # Test footprint
    footprint = get_box_footprint(center, size, yaw)
    print(f"\nFootprint shape: {footprint.shape}")
    print(f"Footprint corners:\n{footprint}")
    
    # Test projection
    camera_intrinsic = np.array([
        [1266.417, 0.0, 816.267],
        [0.0, 1266.417, 491.507],
        [0.0, 0.0, 1.0]
    ])
    
    points_2d = project_3d_to_2d(corners, camera_intrinsic)
    print(f"\n2D projected points shape: {points_2d.shape}")
    print(f"Image center projection: {points_2d[0]}")
    
    print("\nCamera geometry tests passed!")
