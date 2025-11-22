#!/usr/bin/env python3
"""
Video Demo for Self-Driving Perception System
Generates a first-person view video with bounding boxes around detected cars.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class VideoPerceptionDemo:
    """Generate video demo with YOLOv8 car detection on nuScenes dataset."""
    
    def __init__(
        self,
        data_root: str = "./data/nuscenes/v1.0-mini",
        output_dir: str = "./output",
        model_size: str = "n"  # n, s, m, l, x (nano to extra-large)
    ):
        """
        Initialize the video demo.
        
        Args:
            data_root: Path to nuScenes dataset root
            output_dir: Directory to save output videos
            model_size: YOLOv8 model size (n/s/m/l/x)
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLOv8 model (will download pretrained weights on first run)
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f"yolov8{model_size}.pt")
        print("✓ Model loaded")
        
        # COCO relevant class IDs for self-driving perception
        self.detection_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus", 
            7: "truck",
            9: "traffic light",
            11: "stop sign"
        }
        
        # Colors for bounding boxes (BGR format)
        self.colors = {
            0: (255, 100, 100),  # Light blue for persons
            1: (255, 255, 0),    # Cyan for bicycles
            2: (0, 255, 0),      # Green for cars
            3: (255, 0, 255),    # Magenta for motorcycles
            5: (255, 165, 0),    # Orange for buses
            7: (0, 165, 255),    # Orange-red for trucks
            9: (0, 255, 255),    # Yellow for traffic lights
            11: (0, 0, 255)      # Red for stop signs
        }
        
        # Focal length estimation (typical for nuScenes CAM_FRONT)
        # Actual value from calibration is ~1266 pixels
        self.focal_length = 1266.0
        
        # Average real-world car dimensions (meters)
        self.avg_car_width = 1.8
        self.avg_car_height = 1.5
        
    def estimate_distance(
        self,
        bbox_width: float,
        bbox_height: float,
        img_width: int
    ) -> float:
        """
        Estimate distance to detected car using bounding box size.
        
        Args:
            bbox_width: Width of bounding box in pixels
            bbox_height: Height of bounding box in pixels
            img_width: Image width in pixels
            
        Returns:
            Estimated distance in meters
        """
        # Use height-based estimation (more reliable than width)
        if bbox_height > 0:
            distance = (self.avg_car_height * self.focal_length) / bbox_height
        else:
            distance = 50.0  # Default fallback
            
        return distance
    
    def get_distance_color(self, distance: float) -> Tuple[int, int, int]:
        """
        Get color based on distance (traffic light scheme).
        
        Args:
            distance: Distance in meters
            
        Returns:
            BGR color tuple
        """
        if distance < 15:
            return (0, 0, 255)  # Red - close
        elif distance < 30:
            return (0, 165, 255)  # Orange - medium
        else:
            return (0, 255, 0)  # Green - far
    
    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        show_distance: bool = True
    ) -> np.ndarray:
        """
        Process a single frame with YOLOv8 detection.
        
        Args:
            frame: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
            show_distance: Whether to show distance estimates
            
        Returns:
            Annotated frame with bounding boxes
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)[0]
        
        # Get image dimensions
        img_height, img_width = frame.shape[:2]
        
        # Process detections
        annotated_frame = frame.copy()
        car_count = 0
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter for relevant classes and confidence
            if cls_id in self.detection_classes and conf >= conf_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Estimate distance
                distance = self.estimate_distance(bbox_width, bbox_height, img_width)
                
                # Get color based on distance
                if show_distance:
                    color = self.get_distance_color(distance)
                else:
                    color = self.colors.get(cls_id, (255, 255, 255))
                
                # Draw bounding box
                thickness = 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label
                class_name = self.detection_classes[cls_id]
                label = f"{class_name} {conf:.2f}"
                if show_distance:
                    label += f" | {distance:.1f}m"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                # Position label above box (or below if at top of image)
                label_y = y1 - 10 if y1 > 30 else y2 + text_height + 10
                label_x = x1
                
                # Draw label background rectangle
                cv2.rectangle(
                    annotated_frame,
                    (label_x, label_y - text_height - baseline),
                    (label_x + text_width, label_y + baseline),
                    color,
                    -1  # Fill
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (label_x, label_y - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
                
                car_count += 1
        
        # Add info panel at top
        self._add_info_panel(annotated_frame, car_count)
        
        return annotated_frame
    
    def _add_info_panel(self, frame: np.ndarray, car_count: int):
        """Add information panel to frame."""
        panel_height = 40
        
        # Draw semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Detected Objects: {car_count}"
        cv2.putText(frame, text, (10, 25), font, 0.7, (255, 255, 255), 2)
    
    def create_video_from_images(
        self,
        image_paths: List[Path],
        output_name: str = "demo_video.mp4",
        fps: float = 10.0,
        conf_threshold: float = 0.5
    ) -> Path:
        """
        Create video from list of images with detection overlay.
        
        Args:
            image_paths: List of paths to input images
            output_name: Name of output video file
            fps: Frames per second for output video
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Path to created video file
        """
        if not image_paths:
            raise ValueError("No images provided")
        
        output_path = self.output_dir / output_name
        
        # Read first image to get dimensions
        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first image: {image_paths[0]}")
        
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        print(f"Processing {len(image_paths)} frames...")
        
        # Process each frame
        for idx, img_path in enumerate(image_paths):
            # Read frame
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not read {img_path}, skipping")
                continue
            
            # Process with detection
            annotated_frame = self.process_frame(frame, conf_threshold)
            
            # Write to video
            video_writer.write(annotated_frame)
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_paths)} frames")
        
        video_writer.release()
        print(f"✓ Video saved to: {output_path}")
        
        return output_path
    
    def run_nuscenes_scene(
        self,
        scene_name: Optional[str] = None,
        max_frames: Optional[int] = None,
        fps: float = 10.0,
        conf_threshold: float = 0.5
    ) -> Path:
        """
        Run detection on a nuScenes scene.
        
        Args:
            scene_name: Name of scene to process (e.g., "scene-0001")
            max_frames: Maximum number of frames to process (None = all)
            fps: Output video frame rate
            conf_threshold: Detection confidence threshold
            
        Returns:
            Path to output video
        """
        # Check if nuScenes dataset exists
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"nuScenes dataset not found at {self.data_root}\n"
                "Please download v1.0-mini from https://www.nuscenes.org/download"
            )
        
        # Try to use nuscenes-devkit if available
        try:
            from nuscenes.nuscenes import NuScenes
            
            print("Loading nuScenes dataset...")
            nusc = NuScenes(version='v1.0-mini', dataroot=str(self.data_root), verbose=False)
            
            # Get scene
            if scene_name is None:
                scene = nusc.scene[0]
                scene_name = scene['name']
            else:
                scene = next(s for s in nusc.scene if s['name'] == scene_name)
            
            print(f"Processing scene: {scene_name}")
            
            # Collect CAM_FRONT images
            image_paths = []
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = nusc.get('sample', sample_token)
                
                # Get CAM_FRONT data
                cam_front_token = sample['data']['CAM_FRONT']
                cam_front = nusc.get('sample_data', cam_front_token)
                
                # Get image path
                img_path = self.data_root / cam_front['filename']
                image_paths.append(img_path)
                
                # Check max frames limit
                if max_frames and len(image_paths) >= max_frames:
                    break
                
                # Next sample
                sample_token = sample['next']
            
            print(f"Found {len(image_paths)} frames")
            
        except ImportError:
            # Fallback: manually search for CAM_FRONT images
            print("nuscenes-devkit not available, searching for images manually...")
            cam_front_dir = self.data_root / "samples" / "CAM_FRONT"
            
            if not cam_front_dir.exists():
                raise FileNotFoundError(
                    f"Could not find CAM_FRONT images at {cam_front_dir}\n"
                    "Please ensure nuScenes dataset is properly extracted."
                )
            
            image_paths = sorted(cam_front_dir.glob("*.jpg"))[:max_frames]
            scene_name = "manual"
            print(f"Found {len(image_paths)} images")
        
        # Create video
        output_name = f"{scene_name}_detection.mp4"
        return self.create_video_from_images(
            image_paths,
            output_name,
            fps,
            conf_threshold
        )


def main():
    """Main entry point for video demo."""
    print("=" * 60)
    print("Self-Driving Perception - Video Demo")
    print("=" * 60)
    print()
    
    # Create demo instance
    demo = VideoPerceptionDemo(
        data_root="./data/nuscenes/v1.0-mini",
        output_dir="./output",
        model_size="n"  # Use nano model for speed
    )
    
    # Check if dataset exists
    dataset_path = Path("./data/nuscenes/v1.0-mini")
    if not dataset_path.exists():
        print("⚠ nuScenes dataset not found!")
        print()
        print("Please download the mini-nuScenes dataset:")
        print("1. Visit: https://www.nuscenes.org/download")
        print("2. Download: v1.0-mini.tgz (~4GB)")
        print("3. Extract to: ./data/nuscenes/")
        print()
        return
    
    # Run on multiple scenes
    try:
        from nuscenes.nuscenes import NuScenes
        
        print("Loading nuScenes dataset...")
        nusc = NuScenes(version='v1.0-mini', dataroot=str(dataset_path), verbose=False)
        
        # Process all scenes (or limit to first few)
        num_scenes = min(len(nusc.scene), 10)  # Process up to 10 scenes
        print(f"Processing {num_scenes} scenes...")
        print()
        
        all_frames = []
        
        for idx, scene in enumerate(nusc.scene[:num_scenes]):
            scene_name = scene['name']
            print(f"[{idx+1}/{num_scenes}] Processing {scene_name}...")
            
            # Collect CAM_FRONT images for this scene
            sample_token = scene['first_sample_token']
            scene_frames = []
            
            while sample_token:
                sample = nusc.get('sample', sample_token)
                cam_front_token = sample['data']['CAM_FRONT']
                cam_front = nusc.get('sample_data', cam_front_token)
                img_path = dataset_path / cam_front['filename']
                scene_frames.append(img_path)
                sample_token = sample['next']
            
            all_frames.extend(scene_frames)
            print(f"  Added {len(scene_frames)} frames")
        
        print()
        print(f"Total frames: {len(all_frames)}")
        print()
        
        # Create combined video from all scenes
        video_path = demo.create_video_from_images(
            all_frames,
            output_name="multi_scene_detection.mp4",
            fps=10.0,
            conf_threshold=0.5
        )
        
        print()
        print("=" * 60)
        print("DEMO COMPLETE!")
        print(f"Processed {num_scenes} scenes")
        print(f"Total frames: {len(all_frames)}")
        print(f"Output video: {video_path}")
        print("=" * 60)
        
        # Try to open video
        import subprocess
        try:
            subprocess.run(["open", str(video_path)], check=False)
        except:
            pass
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
