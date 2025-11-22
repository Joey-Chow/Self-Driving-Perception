"""
Setup script for the project.
Creates necessary directories and checks dependencies.
"""

import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary project directories."""
    dirs = [
        'data',
        'data/nuscenes',
        'checkpoints',
        'output',
        'configs',
        'logs',
    ]
    
    print("Creating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}/")


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'torch',
        'numpy',
        'opencv-python',
        'matplotlib',
        'nuscenes-devkit',
    ]
    
    optional = [
        'mmdet3d',
        'mmcv',
        'mmdet',
    ]
    
    print("\nChecking dependencies...")
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} NOT FOUND")
            missing.append(package)
    
    print("\nOptional dependencies (for FCOS3D):")
    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  - {package} not installed")
    
    if missing:
        print(f"\n⚠️  Missing required packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required dependencies installed!")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("\n1. Download mini-nuScenes dataset:")
    print("   python scripts/download_nuscenes.py")
    print("\n2. Install MMDetection3D (for FCOS3D model):")
    print("   pip install openmim")
    print("   mim install mmengine")
    print("   mim install 'mmcv>=2.0.0'")
    print("   mim install 'mmdet>=3.0.0'")
    print("   mim install 'mmdet3d>=1.4.0'")
    print("\n3. Download pretrained FCOS3D model:")
    print("   mim download mmdet3d --config fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d --dest ./checkpoints")
    print("\n4. Run inference:")
    print("   python inference.py --scene 0 --visualize --save")
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    print("=" * 70)
    print("3D Car Detection - Project Setup")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Print next steps
    print_next_steps()
    
    if not deps_ok:
        print("\n⚠️  Please install missing dependencies before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()
