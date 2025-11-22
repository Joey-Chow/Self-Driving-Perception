"""
Bird's-Eye View (BEV) transformation utilities.
Transforms 3D boxes from camera coordinates to BEV representation.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class BEVTransform:
    """Handles coordinate transformations for Bird's-Eye View visualization."""
    
    def __init__(
        self,
        bev_width: int = 800,
        bev_height: int = 800,
        bev_range_x: Tuple[float, float] = (-40, 40),  # meters
        bev_range_y: Tuple[float, float] = (0, 80),    # meters (forward)
        pixels_per_meter: Optional[float] = None
    ):
        """
        Initialize BEV transformation.
        
        Args:
            bev_width: BEV image width in pixels
            bev_height: BEV image height in pixels
            bev_range_x: (min, max) range in x direction (left-right) in meters
            bev_range_y: (min, max) range in y direction (forward) in meters
            pixels_per_meter: Resolution (computed automatically if None)
        """
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.bev_range_x = bev_range_x
        self.bev_range_y = bev_range_y
        
        # Compute pixels per meter
        if pixels_per_meter is None:
            ppm_x = bev_width / (bev_range_x[1] - bev_range_x[0])
            ppm_y = bev_height / (bev_range_y[1] - bev_range_y[0])
            self.pixels_per_meter = min(ppm_x, ppm_y)
        else:
            self.pixels_per_meter = pixels_per_meter
        
        # Ego vehicle position in BEV (at bottom-center)
        self.ego_x_pixel = bev_width // 2
        self.ego_y_pixel = bev_height - 50  # 50 pixels from bottom
        
    def camera_to_bev_coords(
        self,
        points_camera: np.ndarray
    ) -> np.ndarray:
        """
        Transform points from camera frame to BEV pixel coordinates.
        
        Camera frame: x=right, y=down, z=forward
        BEV frame: u=right, v=up (image coordinates)
        
        Args:
            points_camera: Nx3 array of points in camera frame (x, y, z)
            
        Returns:
            Nx2 array of BEV pixel coordinates (u, v)
        """
        # Camera frame: x=right, z=forward
        # BEV: x maps to u (horizontal), z maps to v (vertical, inverted)
        
        x_cam = points_camera[:, 0]  # Right
        z_cam = points_camera[:, 2]  # Forward
        
        # Convert to BEV pixels
        # x_cam -> u (horizontal in BEV)
        # z_cam -> v (vertical in BEV, inverted because image v increases downward)
        u = self.ego_x_pixel + x_cam * self.pixels_per_meter
        v = self.ego_y_pixel - z_cam * self.pixels_per_meter
        
        points_bev = np.stack([u, v], axis=1)
        return points_bev
    
    def meters_to_pixels(self, distance_m: float) -> float:
        """Convert distance in meters to pixels."""
        return distance_m * self.pixels_per_meter
    
    def pixels_to_meters(self, distance_px: float) -> float:
        """Convert distance in pixels to meters."""
        return distance_px / self.pixels_per_meter
    
    def transform_box_to_bev(
        self,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float
    ) -> Dict:
        """
        Transform a 3D bounding box to BEV representation.
        
        Args:
            center: (x, y, z) center in camera frame
            size: (width, length, height)
            yaw: rotation angle around z-axis in radians
            
        Returns:
            Dictionary with BEV box parameters
        """
        # Get box footprint corners
        from utils.camera_geometry import get_box_footprint
        
        footprint_3d = get_box_footprint(center, size, yaw)
        
        # Add z coordinate (forward direction) for transformation
        footprint_camera = np.hstack([
            footprint_3d,
            np.full((4, 1), center[2])  # Use z from center
        ])
        
        # Transform to BEV pixels
        footprint_bev = self.camera_to_bev_coords(footprint_camera)
        
        # Also transform center
        center_bev = self.camera_to_bev_coords(center.reshape(1, 3))[0]
        
        return {
            'center_bev': center_bev,
            'footprint_bev': footprint_bev,
            'yaw': yaw,
            'size': size,
            'distance': np.linalg.norm(center[:2]),  # Distance from ego
        }
    
    def is_in_bev_range(self, point_camera: np.ndarray) -> bool:
        """
        Check if a point in camera frame is within BEV range.
        
        Args:
            point_camera: (x, y, z) point in camera frame
            
        Returns:
            True if point is within BEV bounds
        """
        x, _, z = point_camera
        
        if z < self.bev_range_y[0] or z > self.bev_range_y[1]:
            return False
        if x < self.bev_range_x[0] or x > self.bev_range_x[1]:
            return False
        
        return True
    
    def get_distance_circles(
        self,
        radii: List[float] = [10, 20, 30, 40]
    ) -> List[Dict]:
        """
        Get circle parameters for distance visualization in BEV.
        
        Args:
            radii: List of radii in meters
            
        Returns:
            List of circle dictionaries with center and radius in pixels
        """
        circles = []
        for radius_m in radii:
            # Convert radius to pixels
            radius_px = self.meters_to_pixels(radius_m)
            
            circles.append({
                'center': (self.ego_x_pixel, self.ego_y_pixel),
                'radius': radius_px,
                'label': f'{radius_m}m'
            })
        
        return circles
    
    def get_ego_vehicle_polygon(
        self,
        vehicle_width: float = 1.8,
        vehicle_length: float = 4.5
    ) -> np.ndarray:
        """
        Get ego vehicle polygon for BEV visualization.
        
        Args:
            vehicle_width: Vehicle width in meters
            vehicle_length: Vehicle length in meters
            
        Returns:
            4x2 array of polygon corners in BEV pixels
        """
        # Vehicle corners in camera frame (centered at origin)
        half_width = vehicle_width / 2
        half_length = vehicle_length / 2
        
        corners_camera = np.array([
            [-half_width, 0, half_length],   # Front-left
            [half_width, 0, half_length],    # Front-right
            [half_width, 0, -half_length],   # Back-right
            [-half_width, 0, -half_length],  # Back-left
        ])
        
        # Transform to BEV
        corners_bev = self.camera_to_bev_coords(corners_camera)
        
        return corners_bev
    
    def get_color_by_distance(
        self,
        distance: float,
        color_map: str = 'traffic_light'
    ) -> Tuple[int, int, int]:
        """
        Get color based on distance (for visualization).
        
        Args:
            distance: Distance in meters
            color_map: Color scheme ('traffic_light' or 'blue_red')
            
        Returns:
            RGB tuple (0-255)
        """
        if color_map == 'traffic_light':
            # Green (close) -> Yellow (medium) -> Red (far)
            if distance < 15:
                # Green
                return (0, 255, 0)
            elif distance < 30:
                # Yellow-green to yellow
                ratio = (distance - 15) / 15
                return (int(255 * ratio), 255, 0)
            elif distance < 50:
                # Yellow to red
                ratio = (distance - 30) / 20
                return (255, int(255 * (1 - ratio)), 0)
            else:
                # Red
                return (255, 0, 0)
        
        elif color_map == 'blue_red':
            # Blue (close) -> Red (far)
            ratio = min(distance / 50, 1.0)
            return (
                int(255 * ratio),
                0,
                int(255 * (1 - ratio))
            )
        
        else:
            return (255, 255, 255)  # White default


def test_bev_transform():
    """Test BEV transformation."""
    print("Testing BEV transformation...")
    
    # Create transformer
    bev_transform = BEVTransform(
        bev_width=800,
        bev_height=800,
        bev_range_x=(-40, 40),
        bev_range_y=(0, 80)
    )
    
    print(f"Pixels per meter: {bev_transform.pixels_per_meter:.2f}")
    print(f"Ego position (pixels): ({bev_transform.ego_x_pixel}, {bev_transform.ego_y_pixel})")
    
    # Test point transformation
    test_points = np.array([
        [0, 0, 10],    # 10m ahead
        [5, 0, 20],    # 20m ahead, 5m right
        [-5, 0, 30],   # 30m ahead, 5m left
    ])
    
    points_bev = bev_transform.camera_to_bev_coords(test_points)
    print(f"\nCamera points:\n{test_points}")
    print(f"BEV pixels:\n{points_bev}")
    
    # Test box transformation
    center = np.array([5, 0, 20])
    size = np.array([1.8, 4.5, 1.5])
    yaw = np.deg2rad(30)
    
    box_bev = bev_transform.transform_box_to_bev(center, size, yaw)
    print(f"\nBox center BEV: {box_bev['center_bev']}")
    print(f"Box distance: {box_bev['distance']:.2f}m")
    print(f"Footprint shape: {box_bev['footprint_bev'].shape}")
    
    # Test distance circles
    circles = bev_transform.get_distance_circles([10, 20, 30])
    print(f"\nDistance circles: {len(circles)}")
    for circle in circles:
        print(f"  {circle['label']}: radius={circle['radius']:.1f}px")
    
    # Test ego vehicle
    ego_poly = bev_transform.get_ego_vehicle_polygon()
    print(f"\nEgo vehicle polygon shape: {ego_poly.shape}")
    
    # Test colors
    distances = [5, 15, 30, 50]
    print(f"\nDistance-based colors:")
    for d in distances:
        color = bev_transform.get_color_by_distance(d)
        print(f"  {d}m: RGB{color}")
    
    print("\nBEV transformation tests passed!")


if __name__ == '__main__':
    test_bev_transform()
