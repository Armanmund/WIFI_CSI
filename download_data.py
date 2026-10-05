"""
Download the ReWiS dataset from Google Drive.

The dataset is hosted at:
https://drive.google.com/drive/folders/1H-0GFOIHmHpHfdV-T5qv_diov_dCQxMS

This script downloads the 'Formatted_data_frames' folder which contains
pre-processed CSI data from 3 environments (A1, A2, A3) for 4 activities
(walk, empty, jump, stand).

Usage:
    pip install gdown
    python download_data.py
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "few_shot_datasets"


def install_gdown():
    """Install gdown if not already installed."""
    try:
        import gdown
        print("gdown is already installed.")
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("gdown installed successfully.")


def download_dataset():
    """Download the ReWiS formatted data frames from Google Drive."""
    import gdown

    # Google Drive folder ID for Formatted_data_frames
    # From: https://drive.google.com/drive/folders/1H-0GFOIHmHpHfdV-T5qv_diov_dCQxMS
    folder_url = "https://drive.google.com/drive/folders/1H-0GFOIHmHpHfdV-T5qv_diov_dCQxMS"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading ReWiS dataset to: {DATA_DIR}")
    print("This may take a few minutes depending on your connection...\n")

    try:
        gdown.download_folder(
            url=folder_url,
            output=str(DATA_DIR),
            quiet=False,
            use_cookies=False
        )
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 70)
        print(f"""
If the automatic download fails (e.g., due to Google Drive limits),
please download manually:

1. Open this URL in your browser:
   {folder_url}

2. Download the 'Formatted_data_frames' folder contents.
   Look for a folder like 'm1c4_PCA_80_300_extracted_3x4' inside it.

3. Place the data so the structure looks like:
   {DATA_DIR}/
   └── m1c4_PCA_80_300_extracted_3x4/
       ├── train_A1/
       │   ├── walk/       (contains .mat files)
       │   ├── empty/      (contains .mat files)
       │   ├── jump/       (contains .mat files)
       │   └── stand/      (contains .mat files)
       ├── test_A2/
       │   ├── walk/
       │   ├── empty/
       │   ├── jump/
       │   └── stand/
       └── test_A3/
           ├── walk/
           ├── empty/
           ├── jump/
           └── stand/
""")


def verify_dataset():
    """Verify the dataset structure after download."""
    data_folder = "m1c4_PCA_80_300_extracted_3x4"
    expected_envs = {
        "train_A1": ["walk", "empty", "jump", "stand"],
        "test_A2": ["walk", "empty", "jump", "stand"],
        "test_A3": ["walk", "empty", "jump", "stand"],
    }

    print("\n" + "=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)

    all_ok = True
    total_files = 0

    # Search for the data folder in possible locations
    possible_paths = [
        DATA_DIR / data_folder,
        DATA_DIR / "Formatted_data_frames" / data_folder,
        DATA_DIR / "extractd_3x4" / data_folder,
    ]
    
    found_path = None
    for p in possible_paths:
        if p.exists():
            found_path = p
            break
    
    if found_path is None:
        # Search recursively
        for root, dirs, _ in os.walk(DATA_DIR):
            if data_folder in dirs:
                found_path = Path(root) / data_folder
                break
    
    if found_path is None:
        print(f"\n[FAIL] Data folder '{data_folder}' not found under {DATA_DIR}")
        print("Please check the download and folder structure.")
        return False

    print(f"\nFound data at: {found_path}")

    for env_dir, activities in expected_envs.items():
        env_path = found_path / env_dir
        if not env_path.exists():
            print(f"  [FAIL] Missing: {env_dir}/")
            all_ok = False
            continue

        for activity in activities:
            act_path = env_path / activity
            if not act_path.exists():
                print(f"  [FAIL] Missing: {env_dir}/{activity}/")
                all_ok = False
                continue

            mat_files = list(act_path.glob("*.mat"))
            n = len(mat_files)
            total_files += n
            status = "[OK]" if n > 0 else "[FAIL]"
            print(f"  {status} {env_dir}/{activity}: {n} .mat files")
            if n == 0:
                all_ok = False

    print(f"\nTotal .mat files found: {total_files}")

    if all_ok:
        print("\n[OK] Dataset verification PASSED!")
        
        # If data is in a subdirectory, suggest or move it
        expected_direct = DATA_DIR / data_folder
        if found_path != expected_direct:
            print(f"\nNote: Data found at {found_path}")
            print(f"Expected at: {expected_direct}")
            print("You may need to adjust the path in configs/config.py")
    else:
        print("\n[FAIL] Dataset verification FAILED - some files are missing.")

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("ReWiS Dataset Downloader")
    print("Wi-Fi CSI Activity Recognition Dataset")
    print("=" * 60)

    install_gdown()
    download_dataset()
    verify_dataset()
