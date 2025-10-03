"""
🛠️ Utility Functions - Helper functions for Kaggle competitions

Functions:
- setup_colab() - ตั้งค่า Colab environment
- setup_kaggle() - ตั้งค่า Kaggle API
- download_kaggle_dataset() - ดาวน์โหลด dataset จาก Kaggle
- load_data() - โหลดข้อมูล (support csv, parquet, excel, json)
- save_data() - บันทึกข้อมูล
- create_submission() - สร้างไฟล์ submission
- set_seed() - ตั้งค่า random seed
- timer() - Decorator สำหรับวัดเวลา
- reduce_mem_usage() - ลด memory usage (re-export from preprocessing)
- quick_info() - ดูข้อมูลสรุป (re-export from preprocessing)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Any
from functools import wraps
import random

warnings.filterwarnings('ignore')


def setup_colab():
    """
    ตั้งค่า Google Colab environment
    - Mount Google Drive
    - Install required packages
    """
    try:
        from google.colab import drive
        
        print("=" * 60)
        print("🚀 Setting up Google Colab")
        print("=" * 60)
        
        # Mount Drive
        print("\n📁 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Drive mounted at: /content/drive")
        
        # Check if kaggle-utils is installed
        try:
            import kaggle_utils
            print("✅ kaggle-utils already installed")
        except ImportError:
            print("\n📦 Installing kaggle-utils...")
            import subprocess
            subprocess.check_call(['pip', 'install', '-q', 'kaggle-utils'])
            print("✅ kaggle-utils installed")
        
        print("\n" + "=" * 60)
        print("✅ Colab setup complete!")
        print("=" * 60)
        
    except ImportError:
        print("⚠️  Not running in Google Colab")


def setup_kaggle(kaggle_json_path: Optional[str] = None):
    """
    ตั้งค่า Kaggle API
    
    Parameters:
    -----------
    kaggle_json_path : str, optional
        Path to kaggle.json file
        Default: ~/.kaggle/kaggle.json or /content/drive/MyDrive/kaggle.json
    """
    print("=" * 60)
    print("🔑 Setting up Kaggle API")
    print("=" * 60)
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Default paths to check
    if kaggle_json_path is None:
        possible_paths = [
            Path.home() / '.kaggle' / 'kaggle.json',
            Path('/content/drive/MyDrive/kaggle.json'),
            Path('kaggle.json')
        ]
        
        kaggle_json_path = None
        for path in possible_paths:
            if path.exists():
                kaggle_json_path = path
                break
    
    if kaggle_json_path is None:
        print("❌ kaggle.json not found!")
        print("\n📝 To set up:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Place kaggle.json in ~/.kaggle/ or /content/drive/MyDrive/")
        return False
    
    # Copy to .kaggle directory
    import shutil
    target_path = kaggle_dir / 'kaggle.json'
    if not target_path.exists():
        shutil.copy(kaggle_json_path, target_path)
    
    # Set permissions
    os.chmod(target_path, 0o600)
    
    print(f"✅ Kaggle API configured")
    print(f"   Credentials: {target_path}")
    print("=" * 60)
    
    return True


def download_kaggle_dataset(
    competition_name: str,
    download_path: str = './data',
    unzip: bool = True
):
    """
    ดาวน์โหลด dataset จาก Kaggle competition
    
    Parameters:
    -----------
    competition_name : str
        ชื่อ competition (e.g., 'titanic')
    download_path : str, default='./data'
        Path สำหรับบันทึกไฟล์
    unzip : bool, default=True
        แตกไฟล์ zip หรือไม่
    """
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not installed")
        print("Install with: pip install kaggle")
        return
    
    print("=" * 60)
    print(f"📥 Downloading: {competition_name}")
    print("=" * 60)
    
    # Create directory
    os.makedirs(download_path, exist_ok=True)
    
    # Download
    print(f"Downloading to: {download_path}")
    kaggle.api.competition_download_files(
        competition_name,
        path=download_path
    )
    
    if unzip:
        print("\n📦 Extracting files...")
        import zipfile
        zip_file = Path(download_path) / f'{competition_name}.zip'
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_file)
            print("✅ Files extracted")
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print(f"Files in: {download_path}")
    print("=" * 60)
    
    # List files
    files = list(Path(download_path).glob('*'))
    for f in files[:10]:  # Show first 10 files
        print(f"  - {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")


def load_data(
    filepath: str,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    โหลดข้อมูล (รองรับหลาย format)
    
    Parameters:
    -----------
    filepath : str
        Path to file
    verbose : bool, default=True
        แสดงข้อมูล
    **kwargs : dict
        Additional arguments for pandas read functions
    
    Returns:
    --------
    pd.DataFrame
        Data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect file type
    suffix = filepath.suffix.lower()
    
    if verbose:
        print(f"📂 Loading: {filepath.name}")
    
    if suffix == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, **kwargs)
    elif suffix == '.json':
        df = pd.read_json(filepath, **kwargs)
    elif suffix == '.feather':
        df = pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    
    if verbose:
        print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def save_data(
    df: pd.DataFrame,
    filepath: str,
    verbose: bool = True,
    **kwargs
):
    """
    บันทึกข้อมูล
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    filepath : str
        Path to save
    verbose : bool, default=True
        แสดงข้อความ
    **kwargs : dict
        Additional arguments for pandas to_ functions
    """
    filepath = Path(filepath)
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Detect file type
    suffix = filepath.suffix.lower()
    
    if verbose:
        print(f"💾 Saving to: {filepath.name}")
    
    if suffix == '.csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif suffix == '.parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=False, **kwargs)
    elif suffix == '.json':
        df.to_json(filepath, **kwargs)
    elif suffix == '.feather':
        df.to_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    
    if verbose:
        file_size = filepath.stat().st_size / 1024**2
        print(f"✅ Saved: {file_size:.2f} MB")


def create_submission(
    ids: Union[pd.Series, np.ndarray, List],
    predictions: Union[pd.Series, np.ndarray, List],
    filename: str = 'submission.csv',
    id_column: str = 'id',
    target_column: str = 'target',
    verbose: bool = True
) -> pd.DataFrame:
    """
    สร้างไฟล์ submission สำหรับ Kaggle
    
    Parameters:
    -----------
    ids : array-like
        IDs
    predictions : array-like
        Predictions
    filename : str, default='submission.csv'
        ชื่อไฟล์
    id_column : str, default='id'
        ชื่อ column สำหรับ ID
    target_column : str, default='target'
        ชื่อ column สำหรับ predictions
    verbose : bool, default=True
        แสดงข้อความ
    
    Returns:
    --------
    pd.DataFrame
        Submission dataframe
    """
    submission = pd.DataFrame({
        id_column: ids,
        target_column: predictions
    })
    
    submission.to_csv(filename, index=False)
    
    if verbose:
        print("=" * 60)
        print("📝 Submission Created")
        print("=" * 60)
        print(f"Filename: {filename}")
        print(f"Shape: {submission.shape}")
        print(f"\nFirst few rows:")
        print(submission.head())
        print("=" * 60)
    
    return submission


def set_seed(seed: int = 42):
    """
    ตั้งค่า random seed สำหรับ reproducibility
    
    Parameters:
    -----------
    seed : int, default=42
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"🎲 Random seed set to: {seed}")


def timer(func):
    """
    Decorator สำหรับวัดเวลาการทำงานของฟังก์ชัน
    
    Examples:
    ---------
    >>> @timer
    ... def my_function():
    ...     time.sleep(1)
    ...     return "done"
    >>> 
    >>> result = my_function()
    my_function took 1.00 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"⏱️  {func.__name__} took {elapsed:.2f} seconds")
        
        return result
    return wrapper


class Timer:
    """
    Context manager สำหรับวัดเวลา
    
    Examples:
    ---------
    >>> with Timer("Training model"):
    ...     model.fit(X_train, y_train)
    Training model took 12.34 seconds
    """
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"✅ {self.name} took {elapsed:.2f} seconds")


def notify(message: str, sound: bool = False):
    """
    แจ้งเตือนเมื่อโค้ดทำงานเสร็จ
    
    Parameters:
    -----------
    message : str
        ข้อความแจ้งเตือน
    sound : bool, default=False
        เล่นเสียงแจ้งเตือน
    """
    print("\n" + "=" * 60)
    print(f"🔔 NOTIFICATION: {message}")
    print("=" * 60)
    
    if sound:
        try:
            # Try to play a beep sound
            import subprocess
            subprocess.run(['afplay', '/System/Library/Sounds/Glass.aiff'], 
                          capture_output=True)
        except:
            pass  # Silently fail if sound doesn't work


def check_environment() -> dict:
    """
    ตรวจสอบ environment และแพ็คเกจที่ติดตั้ง
    
    Returns:
    --------
    dict
        Environment information
    """
    import sys
    import platform
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
    }
    
    # Check common packages
    packages = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
        'lightgbm', 'xgboost', 'catboost', 'optuna'
    ]
    
    installed = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            installed[pkg] = version
        except ImportError:
            installed[pkg] = 'not installed'
    
    info['packages'] = installed
    
    print("=" * 60)
    print("🔍 Environment Check")
    print("=" * 60)
    print(f"Python: {info['python_version'].split()[0]}")
    print(f"Platform: {info['platform']}")
    print(f"\nPackages:")
    for pkg, version in installed.items():
        status = "✅" if version != 'not installed' else "❌"
        print(f"  {status} {pkg:15s}: {version}")
    print("=" * 60)
    
    return info


def memory_usage():
    """
    แสดง memory usage ปัจจุบัน
    """
    import psutil
    
    process = psutil.Process()
    mem_info = process.memory_info()
    
    print("=" * 60)
    print("💾 Memory Usage")
    print("=" * 60)
    print(f"RSS: {mem_info.rss / 1024**2:.2f} MB")
    print(f"VMS: {mem_info.vms / 1024**2:.2f} MB")
    
    # System memory
    virtual_mem = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total: {virtual_mem.total / 1024**3:.2f} GB")
    print(f"  Available: {virtual_mem.available / 1024**3:.2f} GB")
    print(f"  Used: {virtual_mem.percent}%")
    print("=" * 60)


def estimate_file_size(df: pd.DataFrame) -> str:
    """
    ประมาณขนาดไฟล์ถ้าบันทึกเป็น CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    
    Returns:
    --------
    str
        Estimated file size
    """
    # Rough estimation
    memory_usage = df.memory_usage(deep=True).sum()
    csv_size = memory_usage * 1.5  # CSV usually 1.5x larger than memory
    
    if csv_size < 1024**2:
        return f"{csv_size / 1024:.2f} KB"
    elif csv_size < 1024**3:
        return f"{csv_size / 1024**2:.2f} MB"
    else:
        return f"{csv_size / 1024**3:.2f} GB"


# Re-export common functions from other modules
try:
    from .preprocessing import reduce_mem_usage, quick_info
except ImportError:
    # If running as standalone
    pass


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Set seed
    set_seed(42)
    
    # Timer decorator
    @timer
    def slow_function():
        time.sleep(1)
        return "Done!"
    
    result = slow_function()
    
    # Timer context manager
    with Timer("Loading data"):
        time.sleep(0.5)
    
    # Check environment
    env_info = check_environment()
    
    # Create dummy submission
    ids = range(1, 101)
    predictions = np.random.rand(100)
    submission = create_submission(ids, predictions, verbose=True)