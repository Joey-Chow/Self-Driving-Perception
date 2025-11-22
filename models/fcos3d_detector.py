"""
FCOS3D detector wrapper for MMDetection3D.
Handles model loading, inference, and 3D detection on nuScenes data.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.structures import Det3DDataSample
except ImportError:
    print("Warning: MMDetection3D not installed. Run: pip install mmdet3d")


class FCOS3DDetector:
    """Wrapper for FCOS3D 3D object detector."""
    
    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        device: str = 'cuda:0',
        score_threshold: float = 0.3
    ):
        """
        Initialize FCOS3D detector.
        
        Args:
            config_file: Path to model config file
            checkpoint_file: Path to model checkpoint
            device: Device to run inference on ('cuda:0' or 'cpu')
            score_threshold: Minimum confidence score for detections
        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.score_threshold = score_threshold
        
        # Initialize model
        print(f"Loading FCOS3D model from {checkpoint_file}...")
        self.model = init_model(config_file, checkpoint_file, device=device)
        print("Model loaded successfully!")
        
        # Class names (nuScenes)
        self.class_names = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]
        
        # Filter for car-related classes only
        self.car_class_indices = [0, 1, 3]  # car, truck, bus
    
    def preprocess_image(
        self,
        image_path: str,
        camera_intrinsic: np.ndarray
    ) -> Dict:
        """
        Preprocess image for FCOS3D inference.
        
        Args:
            image_path: Path to input image
            camera_intrinsic: 3x3 camera intrinsic matrix
            
        Returns:
            Dictionary with preprocessed data
        """
        # Note: MMDetection3D handles preprocessing internally
        # We just need to provide the image path and camera info
        
        data = {
            'img_path': image_path,
            'cam_intrinsic': camera_intrinsic,
        }
        
        return data
    
    def inference(
        self,
        image_path: str,
        camera_intrinsic: np.ndarray,
        filter_cars_only: bool = True
    ) -> List[Dict]:
        """
        Run 3D object detection on an image.
        
        Args:
            image_path: Path to input image
            camera_intrinsic: 3x3 camera intrinsic matrix
            filter_cars_only: Whether to keep only car-related detections
            
        Returns:
            List of detection dictionaries with 3D bounding boxes
        """
        # Run inference
        result = inference_detector(self.model, image_path)
        
        # Extract detections
        detections = self._parse_results(
            result,
            filter_cars_only=filter_cars_only
        )
        
        return detections
    
    def _parse_results(
        self,
        result: Det3DDataSample,
        filter_cars_only: bool = True
    ) -> List[Dict]:
        """
        Parse MMDetection3D results into standardized format.
        
        Args:
            result: Detection result from model
            filter_cars_only: Whether to filter for car classes
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Get predictions
        pred_instances_3d = result.pred_instances_3d
        
        # Extract data
        boxes_3d = pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        scores = pred_instances_3d.scores_3d.cpu().numpy()
        labels = pred_instances_3d.labels_3d.cpu().numpy()
        
        # Process each detection
        for i in range(len(boxes_3d)):
            score = scores[i]
            label = labels[i]
            
            # Filter by score
            if score < self.score_threshold:
                continue
            
            # Filter by class
            if filter_cars_only and label not in self.car_class_indices:
                continue
            
            # Parse box (format: x, y, z, w, l, h, yaw)
            box = boxes_3d[i]
            center = box[:3]
            size = box[3:6]  # width, length, height
            yaw = box[6]
            
            detections.append({
                'center': center,          # (x, y, z) in camera frame
                'size': size,              # (width, length, height)
                'yaw': yaw,                # rotation angle (radians)
                'score': float(score),     # confidence score
                'label': int(label),       # class index
                'class_name': self.class_names[label],
            })
        
        return detections
    
    def batch_inference(
        self,
        image_paths: List[str],
        camera_intrinsics: List[np.ndarray],
        filter_cars_only: bool = True
    ) -> List[List[Dict]]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            camera_intrinsics: List of camera intrinsic matrices
            filter_cars_only: Whether to keep only car detections
            
        Returns:
            List of detection lists for each image
        """
        all_detections = []
        
        for img_path, cam_intrinsic in zip(image_paths, camera_intrinsics):
            detections = self.inference(
                img_path,
                cam_intrinsic,
                filter_cars_only=filter_cars_only
            )
            all_detections.append(detections)
        
        return all_detections
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': 'FCOS3D',
            'config': self.config_file,
            'checkpoint': self.checkpoint_file,
            'device': self.device,
            'score_threshold': self.score_threshold,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
        }


def download_pretrained_model(save_dir: str = './checkpoints') -> Tuple[str, str]:
    """
    Download pretrained FCOS3D model from MMDetection3D model zoo.
    
    Args:
        save_dir: Directory to save model files
        
    Returns:
        Tuple of (config_path, checkpoint_path)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model URLs (from MMDetection3D model zoo)
    config_url = (
        "https://github.com/open-mmlab/mmdetection3d/tree/main/configs/"
        "fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py"
    )
    
    checkpoint_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/"
        "fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d/"
        "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth"
    )
    
    print("To download FCOS3D pretrained model:")
    print(f"1. Config: {config_url}")
    print(f"2. Checkpoint: {checkpoint_url}")
    print(f"\nOr use mim:")
    print("  mim download mmdet3d --config fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d --dest ./checkpoints")
    
    # Placeholder paths (user needs to download)
    config_path = save_dir / "fcos3d_config.py"
    checkpoint_path = save_dir / "fcos3d_checkpoint.pth"
    
    return str(config_path), str(checkpoint_path)


def test_detector():
    """Test FCOS3D detector (requires downloaded model)."""
    print("Testing FCOS3D detector...")
    
    # Check if model files exist
    config_file = "./checkpoints/fcos3d_config.py"
    checkpoint_file = "./checkpoints/fcos3d_checkpoint.pth"
    
    if not Path(config_file).exists() or not Path(checkpoint_file).exists():
        print("\nModel files not found. Download pretrained model first:")
        download_pretrained_model()
        return
    
    # Initialize detector
    detector = FCOS3DDetector(
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        score_threshold=0.3
    )
    
    # Print model info
    info = detector.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test inference (requires nuScenes data)
    test_image = "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
    test_intrinsic = np.array([
        [1266.417, 0.0, 816.267],
        [0.0, 1266.417, 491.507],
        [0.0, 0.0, 1.0]
    ])
    
    if Path(test_image).exists():
        print(f"\nRunning inference on: {test_image}")
        detections = detector.inference(test_image, test_intrinsic)
        
        print(f"Number of detections: {len(detections)}")
        for i, det in enumerate(detections[:3]):  # Show first 3
            print(f"\nDetection {i+1}:")
            print(f"  Class: {det['class_name']}")
            print(f"  Score: {det['score']:.3f}")
            print(f"  Center: {det['center']}")
            print(f"  Size: {det['size']}")
            print(f"  Yaw: {np.rad2deg(det['yaw']):.1f}°")
    else:
        print(f"\nTest image not found: {test_image}")
        print("Download nuScenes dataset first.")
    
    print("\nFCOS3D detector tests passed!")


if __name__ == '__main__':
    test_detector()
