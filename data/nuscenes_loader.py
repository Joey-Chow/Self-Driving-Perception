"""
nuScenes dataset loader for 3D car detection.
Handles loading of CAM_FRONT images, calibration, and annotations.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    from pyquaternion import Quaternion
except ImportError:
    print("Warning: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")


class NuScenesLoader:
    """Loader for nuScenes dataset with focus on front camera and car detection."""
    
    def __init__(
        self,
        dataroot: str = './data/nuscenes',
        version: str = 'v1.0-mini',
        camera_channel: str = 'CAM_FRONT',
        car_classes: List[str] = None
    ):
        """
        Initialize nuScenes loader.
        
        Args:
            dataroot: Path to nuScenes dataset root
            version: Dataset version ('v1.0-mini' or 'v1.0-trainval')
            camera_channel: Camera to use (default: CAM_FRONT)
            car_classes: List of vehicle classes to load (default: car only)
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.camera_channel = camera_channel
        
        # Vehicle classes to detect (filter for cars only)
        if car_classes is None:
            self.car_classes = [
                'vehicle.car',
                # Optionally include:
                # 'vehicle.truck',
                # 'vehicle.bus.rigid',
                # 'vehicle.bus.bendy',
            ]
        else:
            self.car_classes = car_classes
        
        # Initialize nuScenes
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
        
        # Cache for faster access
        self._sample_cache = {}
        
    def get_scenes(self) -> List[Dict]:
        """Get list of all scenes in the dataset."""
        return self.nusc.scene
    
    def get_scene_samples(self, scene_token: str) -> List[Dict]:
        """
        Get all samples (keyframes) for a scene.
        
        Args:
            scene_token: Scene identifier
            
        Returns:
            List of sample dictionaries
        """
        scene = self.nusc.get('scene', scene_token)
        samples = []
        
        # Start from first sample
        sample_token = scene['first_sample_token']
        
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']
            
        return samples
    
    def get_camera_data(self, sample_token: str) -> Dict:
        """
        Get camera image and metadata for a sample.
        
        Args:
            sample_token: Sample identifier
            
        Returns:
            Dictionary with image path, calibration, and ego pose
        """
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][self.camera_channel]
        cam_data = self.nusc.get('sample_data', cam_token)
        
        # Get camera calibration
        calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # Get ego pose (vehicle position in world frame)
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Image path
        image_path = self.dataroot / cam_data['filename']
        
        return {
            'image_path': str(image_path),
            'width': cam_data['width'],
            'height': cam_data['height'],
            'timestamp': cam_data['timestamp'],
            'camera_intrinsic': np.array(calib['camera_intrinsic']),
            'camera_translation': np.array(calib['translation']),
            'camera_rotation': Quaternion(calib['rotation']),
            'ego_translation': np.array(ego_pose['translation']),
            'ego_rotation': Quaternion(ego_pose['rotation']),
        }
    
    def get_3d_boxes(
        self,
        sample_token: str,
        filter_classes: bool = True,
        visibility_threshold: int = 1
    ) -> List[Dict]:
        """
        Get 3D bounding boxes for a sample.
        
        Args:
            sample_token: Sample identifier
            filter_classes: Whether to filter for car classes only
            visibility_threshold: Minimum visibility level (1-4, where 4 is fully visible)
            
        Returns:
            List of 3D box dictionaries in camera coordinates
        """
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][self.camera_channel]
        
        # Get boxes in camera frame
        boxes = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Filter by class
            if filter_classes and ann['category_name'] not in self.car_classes:
                continue
            
            # Filter by visibility
            if int(ann['visibility_token']) < visibility_threshold:
                continue
            
            # Get box in global frame
            box = self.nusc.get_box(ann_token)
            
            # Transform to camera frame
            cam_data = self.nusc.get('sample_data', cam_token)
            calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            
            # World -> Ego -> Camera
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            box.translate(-np.array(calib['translation']))
            box.rotate(Quaternion(calib['rotation']).inverse)
            
            # Check if box is in front of camera
            if box.center[2] > 0:  # Positive z is forward
                boxes.append({
                    'center': box.center,  # (x, y, z) in camera coords
                    'size': box.wlh,       # (width, length, height)
                    'rotation': box.orientation,  # Quaternion
                    'yaw': self._quaternion_to_yaw(box.orientation),
                    'velocity': self.nusc.box_velocity(ann_token),
                    'category': ann['category_name'],
                    'instance_token': ann['instance_token'],
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts'],
                })
        
        return boxes
    
    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        """
        Convert quaternion to yaw angle (rotation around z-axis).
        
        Args:
            q: Quaternion
            
        Returns:
            Yaw angle in radians
        """
        # Extract yaw from quaternion
        # yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
        return np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )
    
    def visualize_sample(self, sample_token: str, save_path: Optional[str] = None):
        """
        Visualize a sample with 2D and 3D bounding boxes using nuScenes tools.
        
        Args:
            sample_token: Sample identifier
            save_path: Optional path to save visualization
        """
        self.nusc.render_sample(sample_token, out_path=save_path)
    
    def get_sample_iterator(self, scene_idx: int = 0):
        """
        Get iterator over samples in a scene.
        
        Args:
            scene_idx: Scene index
            
        Yields:
            Tuples of (sample_token, camera_data, boxes_3d)
        """
        scene = self.nusc.scene[scene_idx]
        samples = self.get_scene_samples(scene['token'])
        
        for sample in samples:
            sample_token = sample['token']
            camera_data = self.get_camera_data(sample_token)
            boxes_3d = self.get_3d_boxes(sample_token)
            
            yield sample_token, camera_data, boxes_3d
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        total_samples = len(self.nusc.sample)
        total_scenes = len(self.nusc.scene)
        
        # Count car annotations
        car_count = 0
        for sample in self.nusc.sample:
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                if ann['category_name'] in self.car_classes:
                    car_count += 1
        
        return {
            'version': self.version,
            'total_scenes': total_scenes,
            'total_samples': total_samples,
            'car_annotations': car_count,
            'car_classes': self.car_classes,
            'camera_channel': self.camera_channel,
        }


def test_loader():
    """Test the nuScenes loader."""
    loader = NuScenesLoader(dataroot='./data/nuscenes', version='v1.0-mini')
    
    # Print statistics
    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test loading first sample
    scene = loader.get_scenes()[0]
    samples = loader.get_scene_samples(scene['token'])
    
    print(f"\nScene: {scene['name']}")
    print(f"Number of samples: {len(samples)}")
    
    # Load first sample
    sample_token = samples[0]['token']
    camera_data = loader.get_camera_data(sample_token)
    boxes_3d = loader.get_3d_boxes(sample_token)
    
    print(f"\nFirst sample:")
    print(f"  Image: {camera_data['image_path']}")
    print(f"  Resolution: {camera_data['width']}x{camera_data['height']}")
    print(f"  Number of cars: {len(boxes_3d)}")
    
    if len(boxes_3d) > 0:
        print(f"\nFirst car:")
        box = boxes_3d[0]
        print(f"  Center: {box['center']}")
        print(f"  Size (WxLxH): {box['size']}")
        print(f"  Yaw: {np.rad2deg(box['yaw']):.1f}°")
        print(f"  Category: {box['category']}")


if __name__ == '__main__':
    test_loader()
