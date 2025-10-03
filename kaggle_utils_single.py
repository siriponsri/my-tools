"""
🚀 Kaggle Utils - Single File Version
ชุดเครื่องมือสำหรับมือใหม่หัดแข่ง Kaggle (รวมไว้ในไฟล์เดียว)

Quick Start:
    !wget https://raw.githubusercontent.com/siriponsri/my-tools/main/kaggle_utils_single.py
    from kaggle_utils_single import *

Version: 1.0.0
License: MIT
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Any, Dict

warnings.filterwarnings('ignore')

__version__ = '1.0.0'
__all__ = [
    # Data
    'quick_info',
    'reduce_mem_usage',
    'load_data',
    'save_data',
    
    # Models
    'quick_model_comparison',
    
    # Utils
    'setup_colab',
    'setup_kaggle',
    'create_submission',
    'set_seed',
    
    # Diagnostics
    'check_data_quality',
    'detect_leakage',
    'quick_diagnosis',
]


# ============================================================
# DATA INSPECTION & PREPROCESSING
# ============================================================

def quick_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    แสดงข้อมูลสรุปของ DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการตรวจสอบ
    name : str
        ชื่อของ DataFrame
    
    Returns:
    --------
    pd.DataFrame
        แสดง 5 แถวแรก
    """
    print(f"\n{'='*60}")
    print(f"📊 {name} Information")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n🔍 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percent': missing_pct[missing > 0]
    }).sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values! 🎉")
    
    print(f"\n📈 Data Types:")
    print(df.dtypes.value_counts())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    print(f"\n💾 First few rows:")
    return df.head()


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    ลด memory usage ของ DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการลด memory
    verbose : bool
        แสดงผลลัพธ์หรือไม่
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ที่ลด memory แล้ว
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'💾 Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def load_data(filepath: str, encoding: str = 'utf-8', show_info: bool = False, **kwargs) -> pd.DataFrame:
    """
    โหลดข้อมูลจากไฟล์ (รองรับหลาย format)
    
    Parameters:
    -----------
    filepath : str
        Path ของไฟล์
    encoding : str
        Encoding ('utf-8', 'latin1', 'auto')
    show_info : bool
        แสดงข้อมูลหลัง load
    **kwargs
        ส่งต่อไปยัง pandas.read_*
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ที่โหลดแล้ว
    """
    file_ext = Path(filepath).suffix.lower()
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        elif file_ext in ['.parquet', '.pq']:
            df = pd.read_parquet(filepath, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, **kwargs)
        elif file_ext == '.json':
            df = pd.read_json(filepath, **kwargs)
        elif file_ext == '.feather':
            df = pd.read_feather(filepath, **kwargs)
        elif file_ext in ['.pickle', '.pkl']:
            df = pd.read_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        print(f"✅ Loaded {filepath}")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
        if show_info:
            quick_info(df, Path(filepath).name)
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        raise


def save_data(df: pd.DataFrame, filepath: str, **kwargs):
    """
    บันทึกข้อมูล (auto detect format)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการบันทึก
    filepath : str
        Path ที่จะบันทึก
    **kwargs
        ส่งต่อไปยัง pandas.to_*
    """
    file_ext = Path(filepath).suffix.lower()
    
    try:
        if file_ext == '.csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif file_ext in ['.parquet', '.pq']:
            df.to_parquet(filepath, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            df.to_excel(filepath, index=False, **kwargs)
        elif file_ext == '.json':
            df.to_json(filepath, **kwargs)
        elif file_ext == '.feather':
            df.to_feather(filepath, **kwargs)
        elif file_ext in ['.pickle', '.pkl']:
            df.to_pickle(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        file_size = os.path.getsize(filepath) / 1024**2
        print(f"✅ Saved to {filepath}")
        print(f"   Size: {file_size:.2f} MB")
    
    except Exception as e:
        print(f"❌ Error saving {filepath}: {e}")
        raise


# ============================================================
# MODELS
# ============================================================

def quick_model_comparison(X, y, cv: int = 5, task: str = 'auto', verbose: bool = True):
    """
    เปรียบเทียบหลาย models อย่างรวดเร็ว
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target
    cv : int
        จำนวน CV folds
    task : str
        'auto', 'regression', 'classification'
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    pd.DataFrame
        ผลการเปรียบเทียบ
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, LogisticRegression
    
    # Auto detect task
    if task == 'auto':
        if len(np.unique(y)) <= 20:
            task = 'classification'
        else:
            task = 'regression'
    
    if verbose:
        print(f"🔍 Comparing models ({task})...")
        print(f"{'='*60}")
    
    results = []
    
    if task == 'regression':
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(random_state=42),
        }
        scoring = 'neg_mean_squared_error'
    else:
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        }
        scoring = 'accuracy'
    
    for name, model in models.items():
        if verbose:
            print(f"\n📊 {name}...")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        if task == 'regression':
            scores = np.sqrt(-scores)  # RMSE
        
        results.append({
            'Model': name,
            'Mean': scores.mean(),
            'Std': scores.std(),
            'Min': scores.min(),
            'Max': scores.max()
        })
        
        if verbose:
            print(f"   Score: {scores.mean():.4f} ± {scores.std():.4f}")
    
    results_df = pd.DataFrame(results).sort_values('Mean', ascending=(task=='regression'))
    
    if verbose:
        print(f"\n{'='*60}")
        print("📈 Results Summary:")
        print(results_df.to_string(index=False))
        print(f"\n🏆 Best Model: {results_df.iloc[0]['Model']}")
    
    return results_df


# ============================================================
# UTILS
# ============================================================

def setup_colab():
    """ตั้งค่า Google Colab environment"""
    try:
        from google.colab import drive
        
        print("="*60)
        print("🚀 Setting up Google Colab")
        print("="*60)
        
        print("\n📁 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Drive mounted at: /content/drive")
        
        print("\n" + "="*60)
        print("✅ Colab setup complete!")
        print("="*60)
        
    except ImportError:
        print("⚠️  Not running in Google Colab")


def setup_kaggle(kaggle_json_path: Optional[str] = None):
    """
    ตั้งค่า Kaggle API
    
    Parameters:
    -----------
    kaggle_json_path : str, optional
        Path to kaggle.json file
    """
    print("="*60)
    print("🔑 Setting up Kaggle API")
    print("="*60)
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
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
        print("1. Go to https://www.kaggle.com/")
        print("2. Account → Create New API Token")
        print("3. Download kaggle.json")
        return
    
    import shutil
    target_path = kaggle_dir / 'kaggle.json'
    if kaggle_json_path != target_path:
        shutil.copy(kaggle_json_path, target_path)
    
    os.chmod(target_path, 0o600)
    
    print(f"✅ Kaggle API configured!")
    print(f"   Config file: {target_path}")


def create_submission(
    ids,
    predictions,
    filename: str = 'submission.csv',
    id_column: str = 'id',
    target_column: str = 'target',
    show_sample: bool = True
):
    """
    สร้างไฟล์ submission
    
    Parameters:
    -----------
    ids : array-like
        ID column
    predictions : array-like
        Predictions
    filename : str
        ชื่อไฟล์ที่จะบันทึก
    id_column : str
        ชื่อ column สำหรับ ID
    target_column : str
        ชื่อ column สำหรับ prediction
    show_sample : bool
        แสดง sample หรือไม่
    """
    submission = pd.DataFrame({
        id_column: ids,
        target_column: predictions
    })
    
    submission.to_csv(filename, index=False)
    
    file_size = os.path.getsize(filename) / 1024
    
    print(f"✅ Submission file created: {filename}")
    print(f"📊 Shape: {submission.shape}")
    print(f"💾 Size: {file_size:.2f} KB")
    
    if show_sample:
        print(f"\nSample:")
        print(submission.head())


def set_seed(seed: int = 42):
    """
    ตั้งค่า random seed
    
    Parameters:
    -----------
    seed : int
        Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"✅ Random seed set to {seed}")


# ============================================================
# DIAGNOSTICS
# ============================================================

def check_data_quality(df: pd.DataFrame, target_col: Optional[str] = None, show_details: bool = True):
    """
    ตรวจสอบคุณภาพข้อมูล
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการตรวจสอบ
    target_col : str, optional
        ชื่อ column ของ target
    show_details : bool
        แสดงรายละเอียดหรือไม่
    
    Returns:
    --------
    dict
        รายงานปัญหาและคำแนะนำ
    """
    print("="*70)
    print("🔍 DATA QUALITY CHECK")
    print("="*70)
    
    issues = []
    suggestions = []
    
    # 1. Missing Values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    high_missing = missing_pct[missing_pct > 50]
    
    if len(high_missing) > 0:
        issues.append(f"⚠️  {len(high_missing)} features มี missing > 50%")
        suggestions.append(f"💡 พิจารณาลบ features: {list(high_missing.index[:3])}")
        if show_details:
            print(f"\n📊 High missing values (>50%):")
            print(high_missing.head())
    
    # 2. Constant Features
    constant_features = []
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].nunique() == 1:
            constant_features.append(col)
    
    if constant_features:
        issues.append(f"⚠️  {len(constant_features)} constant features")
        suggestions.append(f"💡 ลบ constant features")
    
    # 3. High Cardinality
    cat_cols = df.select_dtypes(include=['object']).columns
    high_cardinality = []
    for col in cat_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:
            high_cardinality.append(col)
    
    if high_cardinality:
        issues.append(f"⚠️  {len(high_cardinality)} features มี cardinality สูง")
        suggestions.append("💡 พิจารณาใช้ target encoding")
    
    # 4. Target Distribution
    if target_col and target_col in df.columns:
        print(f"\n📊 Target Distribution ({target_col}):")
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
            print(df[target_col].value_counts().head())
        else:
            print(df[target_col].describe())
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📋 SUMMARY")
    print(f"{'='*70}")
    print(f"Issues found: {len(issues)}")
    for issue in issues:
        print(issue)
    
    print(f"\n💡 SUGGESTIONS:")
    for suggestion in suggestions:
        print(suggestion)
    
    return {
        'issues': issues,
        'suggestions': suggestions,
        'high_missing': list(high_missing.index) if len(high_missing) > 0 else [],
        'constant_features': constant_features,
        'high_cardinality': high_cardinality
    }


def detect_leakage(train_df: pd.DataFrame, target_col: str, test_df: Optional[pd.DataFrame] = None):
    """
    ตรวจจับ data leakage
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    target_col : str
        Target column
    test_df : pd.DataFrame, optional
        Test data
    
    Returns:
    --------
    list
        Features ที่น่าสงสัย
    """
    print("="*70)
    print("🔍 DATA LEAKAGE DETECTION")
    print("="*70)
    
    suspicious_features = []
    
    # Check correlation with target
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if target_col in train_df.columns and len(numeric_cols) > 0:
        correlations = train_df[numeric_cols].corrwith(train_df[target_col]).abs()
        high_corr = correlations[correlations > 0.95]
        
        if len(high_corr) > 0:
            print(f"\n⚠️  Features with very high correlation (>0.95):")
            for col, corr in high_corr.items():
                print(f"   {col}: {corr:.4f}")
                suspicious_features.append(col)
    
    # Check train-test differences
    if test_df is not None:
        common_cols = set(train_df.columns) & set(test_df.columns)
        common_cols = [col for col in common_cols if col != target_col]
        
        print(f"\n📊 Train vs Test comparison:")
        for col in list(common_cols)[:5]:  # Check first 5
            if train_df[col].dtype in [np.number]:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                diff_pct = abs(train_mean - test_mean) / train_mean * 100
                
                if diff_pct > 50:
                    print(f"   ⚠️  {col}: {diff_pct:.1f}% difference")
                    suspicious_features.append(col)
    
    if len(suspicious_features) == 0:
        print("\n✅ No obvious leakage detected!")
    else:
        print(f"\n⚠️  Found {len(suspicious_features)} suspicious features")
    
    return list(set(suspicious_features))


def quick_diagnosis(train_df: pd.DataFrame, target_col: str, test_df: Optional[pd.DataFrame] = None):
    """
    วินิจฉัยข้อมูลครบจบในฟังก์ชันเดียว
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    target_col : str
        Target column
    test_df : pd.DataFrame, optional
        Test data
    
    Returns:
    --------
    dict
        รายงานครบถ้วน
    """
    print("\n" + "="*70)
    print("🏥 KAGGLE UTILS - QUICK DIAGNOSIS")
    print("="*70)
    
    # 1. Data Quality
    quality_report = check_data_quality(train_df, target_col)
    
    # 2. Leakage Detection
    if test_df is not None:
        suspicious_features = detect_leakage(train_df, target_col, test_df)
    else:
        suspicious_features = []
    
    # 3. Recommendations
    print("\n" + "="*70)
    print("🎯 RECOMMENDED NEXT STEPS")
    print("="*70)
    
    print("\n1. ✅ Data Cleaning:")
    if quality_report['high_missing']:
        print(f"   - Handle missing values in {len(quality_report['high_missing'])} features")
    if quality_report['constant_features']:
        print(f"   - Remove {len(quality_report['constant_features'])} constant features")
    
    print("\n2. 🔧 Feature Engineering:")
    print("   - Create interaction features")
    print("   - Use target encoding for high cardinality")
    
    print("\n3. 🤖 Model Selection:")
    n_samples = len(train_df)
    n_features = train_df.shape[1]
    
    if n_samples < 10000:
        print("   - Try: Random Forest, LightGBM")
    else:
        print("   - Try: LightGBM, XGBoost (faster)")
    
    if suspicious_features:
        print(f"\n⚠️  WARNING: {len(suspicious_features)} suspicious features detected!")
        print("   Please review before using them in your model.")
    
    return {
        'quality': quality_report,
        'suspicious_features': suspicious_features,
        'n_samples': n_samples,
        'n_features': n_features
    }


# ============================================================
# QUICK START EXAMPLE
# ============================================================

def example_usage():
    """แสดงตัวอย่างการใช้งาน"""
    print("""
    🚀 KAGGLE UTILS - QUICK START
    
    # 1. Load data
    train = load_data('train.csv', show_info=True)
    test = load_data('test.csv')
    
    # 2. Diagnose
    report = quick_diagnosis(train, target_col='target', test_df=test)
    
    # 3. Reduce memory
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    
    # 4. Compare models
    results = quick_model_comparison(X_train, y_train, cv=5)
    
    # 5. Create submission
    create_submission(test_ids, predictions, 'submission.csv')
    
    📚 Full documentation: https://github.com/YOUR_USERNAME/kaggle-utils
    """)


if __name__ == "__main__":
    print(f"Kaggle Utils v{__version__} loaded! 🚀")
    print("Type: example_usage() to see quick start guide")
