"""
Main inference pipeline for 3D car detection.
Integrates data loading, model inference, and BEV visualization.
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

from data.nuscenes_loader import NuScenesLoader
from models.fcos3d_detector import FCOS3DDetector
from visualization.bev_renderer import BEVRenderer


class PerceptionPipeline:
    """End-to-end 3D perception pipeline."""
    
    def __init__(
        self,
        nuscenes_dataroot: str = './data/nuscenes',
        nuscenes_version: str = 'v1.0-mini',
        model_config: str = './checkpoints/fcos3d_config.py',
        model_checkpoint: str = './checkpoints/fcos3d_checkpoint.pth',
        device: str = 'cuda:0',
        score_threshold: float = 0.3,
        bev_width: int = 800,
        bev_height: int = 800
    ):
        """
        Initialize perception pipeline.
        
        Args:
            nuscenes_dataroot: Path to nuScenes dataset
            nuscenes_version: Dataset version
            model_config: Path to model config
            model_checkpoint: Path to model checkpoint
            device: Device for inference
            score_threshold: Detection confidence threshold
            bev_width: BEV image width
            bev_height: BEV image height
        """
        print("Initializing 3D Perception Pipeline...")
        
        # Load dataset
        print("\n[1/3] Loading nuScenes dataset...")
        self.loader = NuScenesLoader(
            dataroot=nuscenes_dataroot,
            version=nuscenes_version,
            camera_channel='CAM_FRONT'
        )
        
        # Print dataset stats
        stats = self.loader.get_statistics()
        print(f"  Dataset: {stats['version']}")
        print(f"  Scenes: {stats['total_scenes']}")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Car annotations: {stats['car_annotations']}")
        
        # Load detector
        print("\n[2/3] Loading FCOS3D model...")
        self.detector = FCOS3DDetector(
            config_file=model_config,
            checkpoint_file=model_checkpoint,
            device=device,
            score_threshold=score_threshold
        )
        
        # Initialize renderer
        print("\n[3/3] Initializing BEV renderer...")
        self.renderer = BEVRenderer(
            bev_width=bev_width,
            bev_height=bev_height,
            bev_range_x=(-40, 40),
            bev_range_y=(0, 80)
        )
        
        print("\n✓ Pipeline initialized successfully!\n")
    
    def process_sample(
        self,
        sample_token: str,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Process a single nuScenes sample.
        
        Args:
            sample_token: Sample identifier
            visualize: Whether to create visualization
            save_path: Optional path to save visualization
            
        Returns:
            Dictionary with detections and visualization
        """
        # Load camera data
        camera_data = self.loader.get_camera_data(sample_token)
        image_path = camera_data['image_path']
        camera_intrinsic = camera_data['camera_intrinsic']
        
        # Run detection
        start_time = time.time()
        detections = self.detector.inference(
            image_path,
            camera_intrinsic,
            filter_cars_only=True
        )
        inference_time = time.time() - start_time
        
        # Get ground truth boxes for comparison
        gt_boxes = self.loader.get_3d_boxes(sample_token)
        
        # Create visualization
        bev_image = None
        if visualize:
            # Prepare info panel
            info = {
                'Detections': str(len(detections)),
                'Ground Truth': str(len(gt_boxes)),
                'Inference': f'{inference_time*1000:.1f}ms',
                'FPS': f'{1.0/inference_time:.1f}'
            }
            
            # Render BEV
            bev_image = self.renderer.render(
                detections=detections,
                info=info,
                show_grid=True,
                show_circles=True
            )
            
            # Save if requested
            if save_path:
                self.renderer.save(bev_image, save_path)
        
        return {
            'sample_token': sample_token,
            'image_path': image_path,
            'detections': detections,
            'ground_truth': gt_boxes,
            'inference_time': inference_time,
            'bev_image': bev_image
        }
    
    def process_scene(
        self,
        scene_idx: int = 0,
        save_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> List[dict]:
        """
        Process all samples in a scene.
        
        Args:
            scene_idx: Scene index
            save_dir: Directory to save visualizations
            max_samples: Maximum number of samples to process
            
        Returns:
            List of result dictionaries
        """
        scene = self.loader.get_scenes()[scene_idx]
        samples = self.loader.get_scene_samples(scene['token'])
        
        print(f"\nProcessing scene: {scene['name']}")
        print(f"Total samples: {len(samples)}")
        
        if max_samples:
            samples = samples[:max_samples]
            print(f"Processing first {max_samples} samples...")
        
        # Prepare save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each sample
        results = []
        for i, sample in enumerate(samples):
            print(f"\rProcessing sample {i+1}/{len(samples)}...", end='')
            
            # Determine save path
            save_path = None
            if save_dir:
                save_path = str(save_dir / f"sample_{i:04d}_bev.png")
            
            # Process sample
            result = self.process_sample(
                sample['token'],
                visualize=True,
                save_path=save_path
            )
            results.append(result)
        
        print("\n✓ Scene processing complete!")
        
        # Compute statistics
        total_detections = sum(len(r['detections']) for r in results)
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        avg_fps = 1.0 / avg_inference_time
        
        print(f"\nStatistics:")
        print(f"  Total detections: {total_detections}")
        print(f"  Avg inference time: {avg_inference_time*1000:.1f}ms")
        print(f"  Avg FPS: {avg_fps:.1f}")
        
        return results
    
    def visualize_result(
        self,
        result: dict,
        show_camera: bool = True,
        show_bev: bool = True,
        window_name: str = "3D Car Detection"
    ):
        """
        Display visualization in window.
        
        Args:
            result: Result dictionary from process_sample
            show_camera: Whether to show camera view
            show_bev: Whether to show BEV
            window_name: OpenCV window name
        """
        images_to_show = []
        
        # Load and annotate camera image
        if show_camera:
            camera_img = cv2.imread(result['image_path'])
            
            # Draw 2D projections (simplified - just show detection count)
            cv2.putText(
                camera_img,
                f"Detections: {len(result['detections'])}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            
            # Resize for display
            camera_img = cv2.resize(camera_img, (800, 450))
            images_to_show.append(camera_img)
        
        # BEV image
        if show_bev and result['bev_image'] is not None:
            images_to_show.append(result['bev_image'])
        
        # Combine images
        if len(images_to_show) == 2:
            combined = np.hstack(images_to_show)
        else:
            combined = images_to_show[0]
        
        # Display
        cv2.imshow(window_name, combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='3D Car Detection for Self-Driving Perception'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        help='Dataset version (v1.0-mini or v1.0-trainval)'
    )
    
    # Model arguments
    parser.add_argument(
        '--config',
        type=str,
        default='./checkpoints/fcos3d_config.py',
        help='Path to model config'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/fcos3d_checkpoint.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for inference (cuda:0 or cpu)'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.3,
        help='Detection confidence threshold'
    )
    
    # Processing arguments
    parser.add_argument(
        '--scene',
        type=int,
        default=0,
        help='Scene index to process'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Specific sample index to process (optional)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization in window'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save visualizations to disk'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PerceptionPipeline(
        nuscenes_dataroot=args.dataroot,
        nuscenes_version=args.version,
        model_config=args.config,
        model_checkpoint=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold
    )
    
    # Process
    if args.sample is not None:
        # Process single sample
        scene = pipeline.loader.get_scenes()[args.scene]
        samples = pipeline.loader.get_scene_samples(scene['token'])
        sample_token = samples[args.sample]['token']
        
        save_path = None
        if args.save:
            save_path = f"{args.output}/sample_{args.sample}_bev.png"
        
        result = pipeline.process_sample(
            sample_token,
            visualize=True,
            save_path=save_path
        )
        
        if args.visualize:
            pipeline.visualize_result(result)
    else:
        # Process scene
        save_dir = args.output if args.save else None
        results = pipeline.process_scene(
            scene_idx=args.scene,
            save_dir=save_dir,
            max_samples=args.max_samples
        )
        
        if args.visualize and len(results) > 0:
            # Show first result
            pipeline.visualize_result(results[0])


if __name__ == '__main__':
    main()
