#!/usr/bin/env python3
"""
Custom 3D Video Visualizer for NuScenes Data
Generates a video of the LiDAR point cloud and Ground Truth bounding boxes.
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
from nuscenes.utils.geometry_utils import view_points

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.nuscenes_loader import NuScenesLoader

def get_box_corners(box):
    """
    Get the 8 corners of a 3D bounding box.
    box: nuscenes.utils.data_classes.Box object
    """
    return box.corners()

def draw_box_3d(ax, corners, color='lime'):
    """
    Draw a 3D bounding box using matplotlib.
    corners: 3x8 numpy array
    """
    # Define the connections between corners to form a box
    # 0-1, 1-2, 2-3, 3-0 (bottom face)
    # 4-5, 5-6, 6-7, 7-4 (top face)
    # 0-4, 1-5, 2-6, 3-7 (vertical lines)
    
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical pillars
    ]
    
    for start, end in connections:
        ax.plot(
            [corners[0, start], corners[0, end]],
            [corners[1, start], corners[1, end]],
            [corners[2, start], corners[2, end]],
            color=color, linewidth=1
        )

def draw_ego_car(ax):
    """Draw the ego vehicle as a solid red box at the origin."""
    # Approximate dimensions of a car (e.g., Renault Zoe)
    l, w, h = 4.0, 1.8, 1.5
    x, y, z = 0, 0, -1.0  # Shift down so LiDAR (0,0,0) is on top
    
    # Define corners
    dx = w / 2
    dy = l / 2
    dz = h / 2
    
    corners = np.array([
        [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz], [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz], # Bottom
        [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz], [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]  # Top
    ])
    
    # Define faces
    faces = [
        [corners[0], corners[1], corners[2], corners[3]], # Bottom
        [corners[4], corners[5], corners[6], corners[7]], # Top
        [corners[0], corners[1], corners[5], corners[4]], # Side
        [corners[2], corners[3], corners[7], corners[6]], # Side
        [corners[1], corners[2], corners[6], corners[5]], # Front
        [corners[4], corners[7], corners[3], corners[0]]  # Back
    ]
    
    # Plot solid box
    ax.add_collection3d(Poly3DCollection(faces, facecolors='red', linewidths=1, edgecolors='darkred', alpha=0.8))

def get_class_info(name):
    """Return mpl_color, bgr_color, and simplified label."""
    name = name.lower()
    if 'human' in name or 'pedestrian' in name:
        return 'red', (0, 0, 255), 'Pedestrian'
    elif 'truck' in name or 'construction' in name:
        return 'orange', (0, 127, 255), 'Truck'
    elif 'bus' in name:
        return 'yellow', (0, 255, 255), 'Bus'
    elif 'car' in name:
        return 'cyan', (255, 255, 0), 'Car'
    elif 'bicycle' in name or 'motorcycle' in name:
        return 'magenta', (255, 0, 255), 'Cycle'
    else:
        return 'lime', (0, 255, 0), 'Object'

def draw_box_3d(ax, corners, color='lime'):
    """
    Draw a 3D bounding box as a solid cuboid.
    corners: 3x8 numpy array
    """
    # Define faces (indices of corners)
    # 0-1-2-3 bottom, 4-5-6-7 top
    faces_indices = [
        [0, 1, 2, 3], # Bottom
        [4, 5, 6, 7], # Top
        [0, 1, 5, 4], # Side 1
        [1, 2, 6, 5], # Side 2
        [2, 3, 7, 6], # Side 3
        [3, 0, 4, 7]  # Side 4
    ]
    
    poly_verts = []
    for indices in faces_indices:
        # corners[:, indices] gives (3, 4) -> transpose to (4, 3)
        poly_verts.append(corners[:, indices].T)
        
    # Render solid transparent box
    ax.add_collection3d(Poly3DCollection(poly_verts, facecolors=color, linewidths=1, edgecolors=color, alpha=0.3))

def render_frame(lidar_points, boxes, output_path=None, dpi=100):
    """
    Render a single 3D frame using Matplotlib.
    """
    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set background color
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Remove axis ticks and labels
    ax.set_axis_off()
    
    # 1. Plot LiDAR Points (Downsampled)
    # Downsample factor (e.g., keep 1 in 20 points) to speed up rendering
    downsample = 20
    points = lidar_points[::downsample]
    
    # Color by INTENSITY (index 3) to show lane lines
    # Intensity is usually 0-255 or 0-100.
    intensities = points[:, 3]
    
    # Normalize intensity for better contrast
    # Clip high values to make lines pop
    intensities = np.clip(intensities, 0, 100)
    
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=intensities, cmap='gray', s=0.5, alpha=0.8, linewidth=0
    )
    
    # 2. Plot Ego Car
    draw_ego_car(ax)
    
    # 3. Plot Bounding Boxes
    for box in boxes:
        corners = get_box_corners(box)
        color, bgr_color, label = get_class_info(box.name)
        draw_box_3d(ax, corners, color=color)
        
        # Add label slightly above the box
        # box.wlh is [w, l, h]
        ax.text(box.center[0], box.center[1], box.center[2] + box.wlh[2]/2 + 0.5, 
                label, color=color, fontsize=6, ha='center')
        
    # 4. Set View
    # Focus on the ego vehicle at (0,0,0)
    limit = 10 # meters
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-10, 10)
    
    # Camera angle
    ax.view_init(elev=45, azim=-45) # Higher elevation to see "lines" better
    
    # 5. Convert to Image
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    
    # Use buffer_rgba() instead of tostring_rgb() which is deprecated/removed in newer matplotlib
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    # Handle Retina display (High DPI) scaling
    # Buffer size might be larger than w*h*4
    if buf.size != w * h * 4:
        scale = int(np.sqrt(buf.size / (w * h * 4)))
        w = w * scale
        h = h * scale
    
    buf = buf.reshape((h, w, 4))
    
    # Convert RGBA to BGR for OpenCV
    image = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return image

def resize_to_width(img, target_width):
    """Resize image to target width while maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h))

def draw_projected_box(img, box, intrinsic, label=None, color=(0, 255, 0)):
    """Draw 3D box projected onto 2D image."""
    corners_3d = box.corners()
    # Project to 2D
    # view_points returns (3, N)
    corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
    corners_2d = corners_2d.astype(int).T
    
    # Draw lines
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4), # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical
    ]
    
    for start, end in connections:
        pt1 = tuple(corners_2d[start])
        pt2 = tuple(corners_2d[end])
        cv2.line(img, pt1, pt2, color, 2)
        
    if label:
        # Find top-left corner for text
        x_min = np.min(corners_2d[:, 0])
        y_min = np.min(corners_2d[:, 1])
        
        # Draw background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x_min, y_min - 20), (x_min + w, y_min), color, -1)
        
        # Draw text (black for contrast)
        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    print("Initializing NuScenes Loader...")
    loader = NuScenesLoader(
        dataroot='./data/nuscenes/v1.0-mini',
        version='v1.0-mini',
        lidar_channel='LIDAR_TOP'
    )
    
    scenes = loader.get_scenes()
    if not scenes:
        print("No scenes found.")
        return
    
    print(f"Found {len(scenes)} scenes.")
    
    # Setup Video Writer
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    video_path = output_dir / "lidar_3d_viz_composite.mp4"
    
    fps = 4 # Slower playback
    writer = None
    
    print("Starting rendering loop...")
    
    # Iterate over first 5 scenes
    for scene in scenes[:5]:
        samples = loader.get_scene_samples(scene['token'])
        print(f"Processing Scene: {scene['name']} ({len(samples)} samples)")
        
        for i, sample in enumerate(tqdm(samples)):
            # 1. Get Data
            lidar_token = sample['data']['LIDAR_TOP']
            cam_front_token = sample['data']['CAM_FRONT']
            
            # Get LiDAR path and boxes
            data_path, boxes, _ = loader.nusc.get_sample_data(lidar_token)
            
            # Get Camera paths and boxes (in camera frame)
            cam_front_path, boxes_front, cam_intrinsic = loader.nusc.get_sample_data(cam_front_token)
            
            # 2. Process LiDAR
            # Read binary file (x, y, z, intensity, ring_index)
            points = np.fromfile(data_path, dtype=np.float32).reshape(-1, 5)
            # Keep x, y, z, intensity
            points_xyzi = points[:, :4]
            
            # 3. Render 3D Frame (Bottom)
            frame_3d = render_frame(points_xyzi, boxes)
            
            # 4. Process Camera (Top)
            img_front = cv2.imread(cam_front_path)
            
            if img_front is None:
                print(f"Warning: Could not load camera images for sample {i}")
                continue
            
            # Draw 3D boxes on Front Camera
            for box in boxes_front:
                _, bgr_color, label = get_class_info(box.name)
                draw_projected_box(img_front, box, cam_intrinsic, label=label, color=bgr_color)
                
            # Resize camera to match 3D frame width
            target_width = frame_3d.shape[1]
            img_front = resize_to_width(img_front, target_width)
            
            # 5. Stack Views: Front (Top), 3D (Bottom)
            composite = np.vstack([img_front, frame_3d])
            
            # Initialize writer if needed
            if writer is None:
                h, w = composite.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                
            writer.write(composite)
        
    if writer:
        writer.release()
        print(f"\nVideo saved to: {video_path}")

if __name__ == "__main__":
    main()
