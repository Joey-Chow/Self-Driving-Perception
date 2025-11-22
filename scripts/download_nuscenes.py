"""
Setup script to download mini-nuScenes dataset.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_nuscenes_mini(data_dir='./data/nuscenes'):
    """
    Download mini-nuScenes dataset.
    
    Note: This script provides download instructions.
    Actual download should be done from the official nuScenes website
    due to license requirements.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("nuScenes Mini Dataset Download Instructions")
    print("=" * 70)
    print("\nTo download the mini-nuScenes dataset:")
    print("\n1. Visit: https://www.nuscenes.org/download")
    print("\n2. Create a free account and accept the terms")
    print("\n3. Download the following files:")
    print("   - v1.0-mini.tgz (Metadata, ~2.5GB)")
    print("   - v1.0-mini/nuScenes-map-expansion-v1.3.zip (Maps, ~130MB)")
    print("\n4. Extract to:", data_dir.absolute())
    print("\n5. Final structure should be:")
    print("   data/nuscenes/")
    print("   ├── maps/")
    print("   ├── samples/")
    print("   │   ├── CAM_FRONT/")
    print("   │   ├── CAM_FRONT_LEFT/")
    print("   │   └── ...")
    print("   ├── sweeps/")
    print("   ├── v1.0-mini/")
    print("   │   ├── attribute.json")
    print("   │   ├── calibrated_sensor.json")
    print("   │   ├── category.json")
    print("   │   └── ...")
    print("\n" + "=" * 70)
    
    # Check if already downloaded
    if (data_dir / 'v1.0-mini').exists():
        print("\n✓ Mini-nuScenes appears to be already downloaded!")
        return True
    
    print("\nAlternatively, use wget/curl:")
    print("\n  # Metadata")
    print("  wget https://www.nuscenes.org/data/v1.0-mini.tgz")
    print("  tar -xzf v1.0-mini.tgz -C data/nuscenes/")
    print("\n  # Note: You'll need to authenticate with the website")
    
    return False


def verify_dataset(data_dir='./data/nuscenes'):
    """Verify dataset is properly downloaded."""
    data_dir = Path(data_dir)
    
    required_dirs = [
        'samples/CAM_FRONT',
        'v1.0-mini',
    ]
    
    required_files = [
        'v1.0-mini/scene.json',
        'v1.0-mini/sample.json',
        'v1.0-mini/sample_data.json',
    ]
    
    print("\nVerifying dataset...")
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} NOT FOUND")
            all_ok = False
    
    for file_path in required_files:
        full_path = data_dir / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} NOT FOUND")
            all_ok = False
    
    if all_ok:
        print("\n✓ Dataset verification passed!")
        return True
    else:
        print("\n✗ Dataset incomplete. Please download missing files.")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download mini-nuScenes dataset')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes',
        help='Directory to download dataset to'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing dataset'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.dataroot)
    else:
        download_nuscenes_mini(args.dataroot)
        verify_dataset(args.dataroot)


if __name__ == '__main__':
    main()
