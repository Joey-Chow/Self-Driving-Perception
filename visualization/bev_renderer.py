"""
Bird's-Eye View (BEV) renderer for 3D car detection visualization.
Creates Tesla-style top-down view with detected vehicles and distance indicators.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from utils.bev_transform import BEVTransform


class BEVRenderer:
    """Renders bird's-eye view visualization of 3D detections."""
    
    def __init__(
        self,
        bev_width: int = 800,
        bev_height: int = 800,
        bev_range_x: Tuple[float, float] = (-40, 40),
        bev_range_y: Tuple[float, float] = (0, 80),
        background_color: Tuple[int, int, int] = (40, 40, 40)
    ):
        """
        Initialize BEV renderer.
        
        Args:
            bev_width: BEV image width in pixels
            bev_height: BEV image height in pixels
            bev_range_x: (min, max) range in x direction (meters)
            bev_range_y: (min, max) range in y direction (meters)
            background_color: RGB background color
        """
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.background_color = background_color
        
        # Initialize BEV transformer
        self.bev_transform = BEVTransform(
            bev_width=bev_width,
            bev_height=bev_height,
            bev_range_x=bev_range_x,
            bev_range_y=bev_range_y
        )
        
    def create_canvas(self) -> np.ndarray:
        """Create blank BEV canvas."""
        canvas = np.full(
            (self.bev_height, self.bev_width, 3),
            self.background_color,
            dtype=np.uint8
        )
        return canvas
    
    def draw_grid(
        self,
        canvas: np.ndarray,
        grid_spacing: float = 10.0,
        color: Tuple[int, int, int] = (60, 60, 60),
        thickness: int = 1
    ):
        """
        Draw grid lines on BEV canvas.
        
        Args:
            canvas: BEV canvas to draw on
            grid_spacing: Grid spacing in meters
            color: RGB color for grid lines
            thickness: Line thickness
        """
        # Vertical lines (constant x)
        x_min, x_max = self.bev_transform.bev_range_x
        y_min, y_max = self.bev_transform.bev_range_y
        
        for x in np.arange(x_min, x_max + grid_spacing, grid_spacing):
            if x == 0:
                continue  # Skip center line (draw separately)
            
            # Transform to pixels
            pt1 = self.bev_transform.camera_to_bev_coords(
                np.array([[x, 0, y_min]])
            )[0]
            pt2 = self.bev_transform.camera_to_bev_coords(
                np.array([[x, 0, y_max]])
            )[0]
            
            cv2.line(
                canvas,
                tuple(pt1.astype(int)),
                tuple(pt2.astype(int)),
                color,
                thickness
            )
        
        # Horizontal lines (constant z/forward)
        for y in np.arange(y_min, y_max + grid_spacing, grid_spacing):
            if y == 0:
                continue
            
            pt1 = self.bev_transform.camera_to_bev_coords(
                np.array([[x_min, 0, y]])
            )[0]
            pt2 = self.bev_transform.camera_to_bev_coords(
                np.array([[x_max, 0, y]])
            )[0]
            
            cv2.line(
                canvas,
                tuple(pt1.astype(int)),
                tuple(pt2.astype(int)),
                color,
                thickness
            )
    
    def draw_distance_circles(
        self,
        canvas: np.ndarray,
        radii: List[float] = [10, 20, 30, 40],
        color: Tuple[int, int, int] = (80, 80, 80),
        thickness: int = 2,
        show_labels: bool = True
    ):
        """
        Draw distance circles around ego vehicle.
        
        Args:
            canvas: BEV canvas to draw on
            radii: List of radii in meters
            color: RGB color for circles
            thickness: Line thickness
            show_labels: Whether to show distance labels
        """
        circles = self.bev_transform.get_distance_circles(radii)
        
        for circle in circles:
            center = circle['center']
            radius = int(circle['radius'])
            label = circle['label']
            
            # Draw circle
            cv2.circle(canvas, center, radius, color, thickness)
            
            # Draw label
            if show_labels:
                label_pos = (center[0] + 5, center[1] - radius + 20)
                cv2.putText(
                    canvas,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 120, 120),
                    1,
                    cv2.LINE_AA
                )
    
    def draw_ego_vehicle(
        self,
        canvas: np.ndarray,
        vehicle_width: float = 1.8,
        vehicle_length: float = 4.5,
        color: Tuple[int, int, int] = (0, 255, 255),  # Cyan
        thickness: int = -1  # Filled
    ):
        """
        Draw ego vehicle at center of BEV.
        
        Args:
            canvas: BEV canvas to draw on
            vehicle_width: Vehicle width in meters
            vehicle_length: Vehicle length in meters
            color: RGB color for vehicle
            thickness: Line thickness (-1 for filled)
        """
        # Get ego vehicle polygon
        ego_poly = self.bev_transform.get_ego_vehicle_polygon(
            vehicle_width, vehicle_length
        )
        
        # Draw polygon
        pts = ego_poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, color, 3)
        
        # Draw direction indicator (forward arrow)
        arrow_start = self.bev_transform.ego_x_pixel, self.bev_transform.ego_y_pixel
        arrow_end = (
            self.bev_transform.ego_x_pixel,
            self.bev_transform.ego_y_pixel - int(vehicle_length * self.bev_transform.pixels_per_meter * 0.8)
        )
        
        cv2.arrowedLine(
            canvas,
            arrow_start,
            arrow_end,
            (255, 255, 255),
            2,
            tipLength=0.3
        )
    
    def draw_3d_box(
        self,
        canvas: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 2,
        show_heading: bool = True
    ):
        """
        Draw a 3D bounding box on BEV.
        
        Args:
            canvas: BEV canvas to draw on
            center: (x, y, z) center in camera frame
            size: (width, length, height)
            yaw: rotation angle in radians
            color: RGB color (auto-computed from distance if None)
            thickness: Line thickness
            show_heading: Whether to show heading direction
        """
        # Transform box to BEV
        box_bev = self.bev_transform.transform_box_to_bev(center, size, yaw)
        
        # Get color based on distance if not specified
        if color is None:
            color = self.bev_transform.get_color_by_distance(
                box_bev['distance'],
                color_map='traffic_light'
            )
        
        # Draw box footprint
        footprint = box_bev['footprint_bev'].astype(np.int32)
        pts = footprint.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, color, thickness)
        
        # Draw heading indicator
        if show_heading:
            center_bev = box_bev['center_bev'].astype(int)
            
            # Calculate heading direction
            heading_length = size[1] * 0.5 * self.bev_transform.pixels_per_meter
            heading_end = (
                int(center_bev[0] + heading_length * np.sin(yaw)),
                int(center_bev[1] - heading_length * np.cos(yaw))
            )
            
            cv2.arrowedLine(
                canvas,
                tuple(center_bev),
                heading_end,
                color,
                thickness,
                tipLength=0.4
            )
    
    def draw_detections(
        self,
        canvas: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_class: bool = True
    ):
        """
        Draw all detected 3D boxes on BEV.
        
        Args:
            canvas: BEV canvas to draw on
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class labels
        """
        for det in detections:
            center = det['center']
            size = det['size']
            yaw = det['yaw']
            
            # Check if in BEV range
            if not self.bev_transform.is_in_bev_range(center):
                continue
            
            # Draw box
            self.draw_3d_box(canvas, center, size, yaw)
            
            # Add text label
            if show_confidence or show_class:
                # Transform center to BEV pixels
                center_bev = self.bev_transform.camera_to_bev_coords(
                    center.reshape(1, 3)
                )[0]
                
                label_parts = []
                if show_class:
                    label_parts.append(det['class_name'])
                if show_confidence:
                    label_parts.append(f"{det['score']:.2f}")
                
                label = ' '.join(label_parts)
                
                # Position label above box
                label_pos = (
                    int(center_bev[0]) - 20,
                    int(center_bev[1]) - int(size[1] * 0.6 * self.bev_transform.pixels_per_meter)
                )
                
                cv2.putText(
                    canvas,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
    
    def add_info_panel(
        self,
        canvas: np.ndarray,
        info: Dict[str, str],
        position: str = 'top-left'
    ):
        """
        Add information panel to BEV canvas.
        
        Args:
            canvas: BEV canvas
            info: Dictionary of info to display
            position: Panel position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        """
        panel_width = 250
        panel_height = 30 + len(info) * 25
        margin = 10
        
        # Determine position
        if position == 'top-left':
            x, y = margin, margin
        elif position == 'top-right':
            x, y = self.bev_width - panel_width - margin, margin
        elif position == 'bottom-left':
            x, y = margin, self.bev_height - panel_height - margin
        else:  # bottom-right
            x, y = self.bev_width - panel_width - margin, self.bev_height - panel_height - margin
        
        # Draw semi-transparent panel
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + panel_width, y + panel_height),
            (20, 20, 20),
            -1
        )
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        # Draw border
        cv2.rectangle(
            canvas,
            (x, y),
            (x + panel_width, y + panel_height),
            (100, 100, 100),
            1
        )
        
        # Draw text
        text_y = y + 20
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                canvas,
                text,
                (x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            text_y += 25
    
    def render(
        self,
        detections: List[Dict],
        info: Optional[Dict[str, str]] = None,
        show_grid: bool = True,
        show_circles: bool = True
    ) -> np.ndarray:
        """
        Render complete BEV visualization.
        
        Args:
            detections: List of 3D detection dictionaries
            info: Optional info to display in panel
            show_grid: Whether to show grid
            show_circles: Whether to show distance circles
            
        Returns:
            Rendered BEV image (numpy array)
        """
        # Create canvas
        canvas = self.create_canvas()
        
        # Draw background elements
        if show_grid:
            self.draw_grid(canvas)
        
        if show_circles:
            self.draw_distance_circles(canvas)
        
        # Draw detections
        self.draw_detections(canvas, detections)
        
        # Draw ego vehicle (on top)
        self.draw_ego_vehicle(canvas)
        
        # Add info panel
        if info is not None:
            self.add_info_panel(canvas, info, position='top-left')
        
        return canvas
    
    def save(self, canvas: np.ndarray, save_path: str):
        """Save BEV image to file."""
        cv2.imwrite(save_path, canvas)
        print(f"Saved BEV visualization to: {save_path}")


def test_renderer():
    """Test BEV renderer."""
    print("Testing BEV renderer...")
    
    # Create renderer
    renderer = BEVRenderer()
    
    # Create test detections
    test_detections = [
        {
            'center': np.array([0, 0, 15]),
            'size': np.array([1.8, 4.5, 1.5]),
            'yaw': 0,
            'score': 0.95,
            'class_name': 'car'
        },
        {
            'center': np.array([5, 0, 25]),
            'size': np.array([1.8, 4.5, 1.5]),
            'yaw': np.deg2rad(30),
            'score': 0.87,
            'class_name': 'car'
        },
        {
            'center': np.array([-8, 0, 35]),
            'size': np.array([2.2, 6.0, 2.5]),
            'yaw': np.deg2rad(-15),
            'score': 0.78,
            'class_name': 'truck'
        },
    ]
    
    # Test info
    test_info = {
        'Detections': '3',
        'FPS': '15.2',
        'Range': '80m'
    }
    
    # Render
    bev_image = renderer.render(
        detections=test_detections,
        info=test_info,
        show_grid=True,
        show_circles=True
    )
    
    print(f"Rendered BEV image shape: {bev_image.shape}")
    
    # Save test image
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    renderer.save(bev_image, './output/test_bev.png')
    
    print("\nBEV renderer tests passed!")


if __name__ == '__main__':
    test_renderer()
