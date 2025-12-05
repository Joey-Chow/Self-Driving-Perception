"""
nuScenes dataset loader for 3D perception (Camera + LiDAR).
Handles loading of synchronized Camera and LiDAR data with calibration matrices.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    from pyquaternion import Quaternion
except ImportError:
    print("Warning: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")


class NuScenesLoader:
    """Loader for nuScenes dataset supporting both Camera and LiDAR data."""
    
    def __init__(
        self,
        dataroot: str = './data/nuscenes',
        version: str = 'v1.0-mini',
        camera_channel: str = 'CAM_FRONT',
        lidar_channel: str = 'LIDAR_TOP',
        car_classes: List[str] = None
    ):
        self.dataroot = Path(dataroot)
        self.version = version
        self.camera_channel = camera_channel
        self.lidar_channel = lidar_channel
        
        # Vehicle classes to detect
        if car_classes is None:
            self.car_classes = [
                'vehicle.car', 'vehicle.truck', 'vehicle.bus.rigid', 'vehicle.bus.bendy'
            ]
        else:
            self.car_classes = car_classes
        
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
        
    def get_scenes(self) -> List[Dict]:
        return self.nusc.scene
    
    def get_scene_samples(self, scene_token: str) -> List[Dict]:
        """Get all samples (keyframes) for a scene."""
        scene = self.nusc.get('scene', scene_token)
        samples = []
        sample_token = scene['first_sample_token']
        
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']
            
        return samples
    
    def get_multimodal_data(self, sample_token: str) -> Dict:
        """
        [CRITICAL FOR PHASE 2]
        Get synchronized Camera AND LiDAR data paths, plus calibration matrices.
        
        Returns:
            Dictionary containing:
            - image_path, lidar_path
            - lidar2ego_translation, lidar2ego_rotation
            - cam2ego_translation, cam2ego_rotation
            - camera_intrinsic
        """
        sample = self.nusc.get('sample', sample_token)
        
        # 1. Get Camera Data
        cam_token = sample['data'][self.camera_channel]
        cam_data = self.nusc.get('sample_data', cam_token)
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # 2. Get LiDAR Data
        lidar_token = sample['data'][self.lidar_channel]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Paths
        image_path = self.dataroot / cam_data['filename']
        lidar_path = self.dataroot / lidar_data['filename']
        
        return {
            'token': sample_token,
            # Paths
            'image_path': str(image_path),
            'lidar_path': str(lidar_path),
            
            # LiDAR Calibration (LiDAR -> Ego)
            'lidar2ego_translation': lidar_calib['translation'],
            'lidar2ego_rotation': lidar_calib['rotation'],
            
            # Camera Calibration (Camera -> Ego)
            'cam2ego_translation': cam_calib['translation'],
            'cam2ego_rotation': cam_calib['rotation'],
            'camera_intrinsic': np.array(cam_calib['camera_intrinsic']),
            
            # Ego Pose (Global coordinates, useful for accumulation)
            'ego2global_translation': cam_ego_pose['translation'],
            'ego2global_rotation': cam_ego_pose['rotation'],
            
            # Timestamps (for synchronization checks)
            'timestamp_cam': cam_data['timestamp'],
            'timestamp_lidar': lidar_data['timestamp']
        }

    def get_lidar2cam_matrix(self, data_dict: Dict) -> np.ndarray:
        """
        Helper: Compute the 4x4 transformation matrix from LiDAR to Camera.
        Math: T_lidar2cam = inv(T_cam2ego) @ T_lidar2ego
        """
        # 1. Form T_lidar2ego
        t_l2e = transform_matrix(
            data_dict['lidar2ego_translation'],
            Quaternion(data_dict['lidar2ego_rotation']),
            inverse=False
        )
        
        # 2. Form T_cam2ego
        t_c2e = transform_matrix(
            data_dict['cam2ego_translation'],
            Quaternion(data_dict['cam2ego_rotation']),
            inverse=False
        )
        
        # 3. T_lidar2cam = inv(T_cam2ego) @ T_lidar2ego
        t_l2c = np.linalg.inv(t_c2e) @ t_l2e
        
        return t_l2c
    
    def get_3d_boxes(self, sample_token: str) -> List[Dict]:
        """Get GT boxes (unchanged logic, returns boxes in global/sensor frame)."""
        # for simplicity, this shows how nuScenes standard method is used to get boxes under LiDAR coordinates system
        
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data'][self.lidar_channel]
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        
        # boxes already in LiDAR coordinates system
        return boxes 


def test_loader():
    """Test Phase 2 features."""
    loader = NuScenesLoader(dataroot='./data/nuscenes', version='v1.0-mini')
    
    scene = loader.get_scenes()[0]
    samples = loader.get_scene_samples(scene['token'])
    first_sample = samples[0]
    
    print(f"Testing Phase 2 Data Loading...")
    data = loader.get_multimodal_data(first_sample['token'])
    
    print(f"Image: {data['image_path']}")
    print(f"LiDAR: {data['lidar_path']}")
    
    # Calculate projection matrix
    l2c_mat = loader.get_lidar2cam_matrix(data)
    print(f"LiDAR to Camera Matrix:\n{l2c_mat.round(2)}")
    
    print("\n✓ Loader is ready for Phase 2!")

if __name__ == '__main__':
    test_loader()