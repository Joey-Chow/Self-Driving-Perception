# models/lidar_detector.py

from mmdet3d.apis import init_model, inference_detector
import numpy as np


class PointPillarsDetector:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        """
        Initialize PointPillars model
        Args:
            config_path: model configuration file (.py)
            checkpoint_path: pre-trained weight (.pth)
        """
        print("Loading PointPillars model...")
        self.model = init_model(config_path, checkpoint_path, device=device)
        print("✓ PointPillars model loaded")

    def detect(self, pcd_path):
        """
        Run deduction
        Args:
            pcd_path: .bin Lidar file path
        Returns:
            results: includes 3D boxes (x, y, z, l, w, h, rot) results
        """
        # inference_detector supports direct transmission file path
        result, _ = inference_detector(self.model, pcd_path)

        # Analysis result (Adjust analysis logic here using mmdet3d version)
        # typically result.pred_instances_3d.bboxes_3d is 3d boxes
        pred_instances = result.pred_instances_3d
        bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
        scores_3d = pred_instances.scores_3d.cpu().numpy()
        labels_3d = pred_instances.labels_3d.cpu().numpy()

        return bboxes_3d, scores_3d, labels_3d
