"""
Outlier Detection and Handling
เครื่องมือสำหรับตรวจจับและจัดการ outliers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== DETECTION METHODS ====================

def detect_outliers_iqr(df, columns=None, threshold=1.5, verbose=True):
    """
    ตรวจจับ outliers ด้วย IQR (Interquartile Range) method
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการตรวจสอบ
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ (None = numeric ทั้งหมด)
    threshold : float
        IQR threshold (1.5 = standard, 3.0 = extreme outliers)
    verbose : bool
        แสดง progress bar
    
    Returns:
    --------
    list
        Index ของ rows ที่เป็น outliers
    
    Example:
    --------
    >>> outliers = detect_outliers_iqr(train, ['price', 'area'], threshold=1.5)
    >>> print(f"Found {len(outliers)} outliers")
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_indices = []
    outlier_details = {}
    
    print(f"🔍 Detecting outliers using IQR method (threshold={threshold})")
    print(f"{'='*60}")
    
    col_iterator = tqdm(
        columns,
        desc="Checking columns",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for col in col_iterator:
        col_iterator.set_postfix({'Column': col[:20]})
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        column_outliers = df[outlier_mask].index.tolist()
        
        if column_outliers:
            outlier_indices.extend(column_outliers)
            outlier_details[col] = {
                'count': len(column_outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'indices': column_outliers
            }
    
    # Get unique outliers
    outlier_indices = list(set(outlier_indices))
    
    print(f"\n📊 Results:")
    print(f"   Total outliers found: {len(outlier_indices)}")
    print(f"   Affected rows: {len(outlier_indices) / len(df) * 100:.2f}%")
    
    if outlier_details and verbose:
        print(f"\n📋 Details by column:")
        for col, details in sorted(outlier_details.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            print(f"   {col}: {details['count']} outliers")
    
    return outlier_indices


def detect_outliers_zscore(df, columns=None, threshold=3, verbose=True):
    """
    ตรวจจับ outliers ด้วย Z-score method
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการตรวจสอบ
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ
    threshold : float
        Z-score threshold (3 = standard, 2 = more sensitive)
    verbose : bool
        แสดง progress bar
    
    Returns:
    --------
    list
        Index ของ rows ที่เป็น outliers
    
    Example:
    --------
    >>> outliers = detect_outliers_zscore(train, ['price', 'area'], threshold=3)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_indices = []
    outlier_details = {}
    
    print(f"🔍 Detecting outliers using Z-score method (threshold={threshold})")
    print(f"{'='*60}")
    
    col_iterator = tqdm(
        columns,
        desc="Checking columns",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for col in col_iterator:
        col_iterator.set_postfix({'Column': col[:20]})
        
        mean = df[col].mean()
        std = df[col].std()
        
        z_scores = np.abs((df[col] - mean) / std)
        outlier_mask = z_scores > threshold
        column_outliers = df[outlier_mask].index.tolist()
        
        if column_outliers:
            outlier_indices.extend(column_outliers)
            outlier_details[col] = {
                'count': len(column_outliers),
                'mean': mean,
                'std': std,
                'indices': column_outliers
            }
    
    outlier_indices = list(set(outlier_indices))
    
    print(f"\n📊 Results:")
    print(f"   Total outliers found: {len(outlier_indices)}")
    print(f"   Affected rows: {len(outlier_indices) / len(df) * 100:.2f}%")
    
    if outlier_details and verbose:
        print(f"\n📋 Details by column:")
        for col, details in sorted(outlier_details.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            print(f"   {col}: {details['count']} outliers")
    
    return outlier_indices


def detect_outliers_isolation_forest(df, columns=None, contamination=0.1, 
                                     random_state=42, verbose=True):
    """
    ตรวจจับ outliers ด้วย Isolation Forest
    ดีสำหรับ multivariate outliers
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการตรวจสอบ
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ
    contamination : float
        สัดส่วนของ outliers ที่คาดว่ามี (0.1 = 10%)
    random_state : int
        Random seed
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    list
        Index ของ rows ที่เป็น outliers
    
    Example:
    --------
    >>> outliers = detect_outliers_isolation_forest(
    ...     train, ['price', 'area'], 
    ...     contamination=0.1
    ... )
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"🔍 Detecting outliers using Isolation Forest")
    print(f"{'='*60}")
    print(f"Contamination: {contamination}")
    print(f"Features: {len(columns)}")
    
    if verbose:
        print(f"\n🌲 Training Isolation Forest...")
    
    X = df[columns].fillna(df[columns].mean())
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Predict (-1 = outlier, 1 = inlier)
    predictions = iso_forest.fit_predict(X)
    outlier_indices = df[predictions == -1].index.tolist()
    
    print(f"\n📊 Results:")
    print(f"   Total outliers found: {len(outlier_indices)}")
    print(f"   Affected rows: {len(outlier_indices) / len(df) * 100:.2f}%")
    
    return outlier_indices


def detect_outliers_lof(df, columns=None, n_neighbors=20, contamination=0.1, verbose=True):
    """
    ตรวจจับ outliers ด้วย Local Outlier Factor (LOF)
    ดีสำหรับ local density-based outliers
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ
    n_neighbors : int
        จำนวน neighbors ที่ใช้คำนวณ
    contamination : float
        สัดส่วนของ outliers
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    list
        Index ของ rows ที่เป็น outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"🔍 Detecting outliers using Local Outlier Factor (LOF)")
    print(f"{'='*60}")
    print(f"Neighbors: {n_neighbors}")
    print(f"Contamination: {contamination}")
    
    if verbose:
        print(f"\n🎯 Computing local outlier factors...")
    
    X = df[columns].fillna(df[columns].mean())
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )
    
    predictions = lof.fit_predict(X)
    outlier_indices = df[predictions == -1].index.tolist()
    
    print(f"\n📊 Results:")
    print(f"   Total outliers found: {len(outlier_indices)}")
    print(f"   Affected rows: {len(outlier_indices) / len(df) * 100:.2f}%")
    
    return outlier_indices


def detect_outliers_elliptic(df, columns=None, contamination=0.1, random_state=42, verbose=True):
    """
    ตรวจจับ outliers ด้วย Elliptic Envelope
    ดีสำหรับ Gaussian distributed data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ
    contamination : float
        สัดส่วนของ outliers
    random_state : int
        Random seed
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    list
        Index ของ rows ที่เป็น outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"🔍 Detecting outliers using Elliptic Envelope")
    print(f"{'='*60}")
    print(f"Contamination: {contamination}")
    
    if verbose:
        print(f"\n🎯 Fitting elliptic envelope...")
    
    X = df[columns].fillna(df[columns].mean())
    
    try:
        envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state
        )
        predictions = envelope.fit_predict(X)
        outlier_indices = df[predictions == -1].index.tolist()
        
        print(f"\n📊 Results:")
        print(f"   Total outliers found: {len(outlier_indices)}")
        print(f"   Affected rows: {len(outlier_indices) / len(df) * 100:.2f}%")
        
        return outlier_indices
    except Exception as e:
        print(f"\n⚠️  Error: {str(e)}")
        print(f"   Try using other methods or check data distribution")
        return []


# ==================== ENSEMBLE DETECTION ====================

def detect_outliers_ensemble(df, columns=None, methods=['iqr', 'zscore', 'isolation'],
                            min_votes=2, verbose=True):
    """
    ตรวจจับ outliers โดยใช้หลายวิธีร่วมกัน (Ensemble)
    outlier ต้องถูกตรวจพบจาก >= min_votes methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ
    methods : list
        Methods ที่จะใช้: 'iqr', 'zscore', 'isolation', 'lof', 'elliptic'
    min_votes : int
        จำนวน methods ขั้นต่ำที่ต้องตรวจพบ
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    dict
        {'outliers': list, 'votes': dict}
    
    Example:
    --------
    >>> result = detect_outliers_ensemble(
    ...     train, ['price', 'area'],
    ...     methods=['iqr', 'zscore', 'isolation'],
    ...     min_votes=2
    ... )
    >>> outliers = result['outliers']
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"{'='*60}")
    print(f"🎯 Ensemble Outlier Detection")
    print(f"{'='*60}")
    print(f"Methods: {methods}")
    print(f"Min votes required: {min_votes}")
    print(f"Columns: {len(columns)}")
    
    all_outliers = {}
    
    # Run each method
    method_iterator = tqdm(
        methods,
        desc="Running methods",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for method in method_iterator:
        method_iterator.set_postfix({'Method': method})
        
        if method == 'iqr':
            outliers = detect_outliers_iqr(df, columns, verbose=False)
        elif method == 'zscore':
            outliers = detect_outliers_zscore(df, columns, verbose=False)
        elif method == 'isolation':
            outliers = detect_outliers_isolation_forest(df, columns, verbose=False)
        elif method == 'lof':
            outliers = detect_outliers_lof(df, columns, verbose=False)
        elif method == 'elliptic':
            outliers = detect_outliers_elliptic(df, columns, verbose=False)
        else:
            print(f"⚠️  Unknown method: {method}")
            continue
        
        all_outliers[method] = set(outliers)
    
    # Count votes
    from collections import Counter
    all_indices = []
    for outliers_set in all_outliers.values():
        all_indices.extend(list(outliers_set))
    
    vote_counts = Counter(all_indices)
    
    # Filter by min_votes
    ensemble_outliers = [idx for idx, votes in vote_counts.items() if votes >= min_votes]
    
    print(f"\n{'='*60}")
    print(f"📊 Ensemble Results:")
    print(f"{'='*60}")
    for method, outliers_set in all_outliers.items():
        print(f"   {method:15s}: {len(outliers_set):5d} outliers")
    print(f"   {'─'*40}")
    print(f"   {'Ensemble':15s}: {len(ensemble_outliers):5d} outliers (≥{min_votes} votes)")
    
    return {
        'outliers': ensemble_outliers,
        'votes': vote_counts,
        'by_method': {k: list(v) for k, v in all_outliers.items()}
    }


# ==================== HANDLING METHODS ====================

def handle_outliers(df, columns=None, method='cap', iqr_threshold=1.5, verbose=True):
    """
    จัดการ outliers ด้วยวิธีต่างๆ
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    columns : list, optional
        Columns ที่ต้องการจัดการ
    method : str
        'cap' (winsorize), 'remove', 'log', 'sqrt', 'clip'
    iqr_threshold : float
        IQR threshold สำหรับ cap/clip
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ที่จัดการ outliers แล้ว
    
    Example:
    --------
    >>> # Cap outliers
    >>> df_clean = handle_outliers(train, ['price', 'area'], method='cap')
    >>> 
    >>> # Remove outliers
    >>> df_clean = handle_outliers(train, ['price', 'area'], method='remove')
    >>> 
    >>> # Transform with log
    >>> df_clean = handle_outliers(train, ['price', 'area'], method='log')
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    print(f"{'='*60}")
    print(f"🔧 Handling Outliers")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"Columns: {len(columns)}")
    
    if method in ['cap', 'clip']:
        col_iterator = tqdm(
            columns,
            desc="Processing",
            disable=not verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for col in col_iterator:
            col_iterator.set_postfix({'Column': col[:20]})
            
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_threshold * IQR
            upper_bound = Q3 + iqr_threshold * IQR
            
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        print(f"✅ Outliers capped using IQR method")
        
    elif method == 'remove':
        original_len = len(df_clean)
        
        # Detect outliers
        outliers = detect_outliers_iqr(df_clean, columns, iqr_threshold, verbose=False)
        
        # Remove
        df_clean = df_clean.drop(outliers)
        df_clean = df_clean.reset_index(drop=True)
        
        print(f"✅ Removed {len(outliers)} rows ({len(outliers)/original_len*100:.2f}%)")
        
    elif method == 'log':
        col_iterator = tqdm(
            columns,
            desc="Transforming",
            disable=not verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for col in col_iterator:
            col_iterator.set_postfix({'Column': col[:20]})
            
            # Add small constant if there are zeros/negatives
            min_val = df_clean[col].min()
            if min_val <= 0:
                df_clean[col] = df_clean[col] - min_val + 1
            
            df_clean[col] = np.log1p(df_clean[col])
        
        print(f"✅ Applied log transformation")
        
    elif method == 'sqrt':
        col_iterator = tqdm(
            columns,
            desc="Transforming",
            disable=not verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for col in col_iterator:
            col_iterator.set_postfix({'Column': col[:20]})
            
            # Ensure non-negative
            min_val = df_clean[col].min()
            if min_val < 0:
                df_clean[col] = df_clean[col] - min_val
            
            df_clean[col] = np.sqrt(df_clean[col])
        
        print(f"✅ Applied square root transformation")
    
    else:
        print(f"⚠️  Unknown method: {method}")
        print(f"   Available: 'cap', 'remove', 'log', 'sqrt', 'clip'")
        return df
    
    print(f"\n📊 Final shape: {df_clean.shape}")
    return df_clean


# ==================== VISUALIZATION ====================

def plot_outliers(df, column, method='iqr', threshold=1.5, figsize=(15, 5)):
    """
    Plot outliers พร้อมแสดงตำแหน่งของ outliers
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    column : str
        Column ที่ต้องการ plot
    method : str
        'iqr' หรือ 'zscore'
    threshold : float
        Threshold สำหรับตรวจจับ outliers
    figsize : tuple
        Figure size
    
    Example:
    --------
    >>> plot_outliers(train, 'price', method='iqr')
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    data = df[column].dropna()
    
    # Detect outliers
    if method == 'iqr':
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_mask = (data < lower_bound) | (data > upper_bound)
    else:  # zscore
        mean, std = data.mean(), data.std()
        z_scores = np.abs((data - mean) / std)
        outliers_mask = z_scores > threshold
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    outliers = data[outliers_mask]
    inliers = data[~outliers_mask]
    
    # 1. Boxplot
    axes[0].boxplot(data, vert=True)
    axes[0].set_title(f'Boxplot: {column}')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[1].hist(inliers, bins=50, alpha=0.7, color='blue', label='Inliers', edgecolor='black')
    axes[1].hist(outliers, bins=20, alpha=0.7, color='red', label='Outliers', edgecolor='black')
    axes[1].axvline(lower_bound, color='red', linestyle='--', linewidth=2, label=f'Bounds')
    axes[1].axvline(upper_bound, color='red', linestyle='--', linewidth=2)
    axes[1].set_title(f'Distribution: {column}')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Scatter plot (index vs value)
    axes[2].scatter(inliers.index, inliers, alpha=0.5, s=10, color='blue', label='Inliers')
    axes[2].scatter(outliers.index, outliers, alpha=0.8, s=20, color='red', label='Outliers')
    axes[2].axhline(lower_bound, color='red', linestyle='--', linewidth=2)
    axes[2].axhline(upper_bound, color='red', linestyle='--', linewidth=2)
    axes[2].set_title(f'Scatter: {column}')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Summary
    print(f"📊 Outlier Analysis: {column}")
    print(f"{'='*60}")
    print(f"Method: {method} (threshold={threshold})")
    print(f"Total values: {len(data)}")
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Min outlier: {outliers.min():.2f}" if len(outliers) > 0 else "Min outlier: N/A")
    print(f"Max outlier: {outliers.max():.2f}" if len(outliers) > 0 else "Max outlier: N/A")
    
    plt.show()


def plot_outliers_comparison(df, column, figsize=(15, 5)):
    """
    เปรียบเทียบ outlier detection methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    column : str
        Column ที่ต้องการเปรียบเทียบ
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    data = df[column].dropna()
    
    methods_info = []
    
    # IQR
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    iqr_lower = Q1 - 1.5 * IQR
    iqr_upper = Q3 + 1.5 * IQR
    iqr_outliers = data[(data < iqr_lower) | (data > iqr_upper)]
    methods_info.append(('IQR', iqr_outliers, iqr_lower, iqr_upper))
    
    # Z-score
    mean, std = data.mean(), data.std()
    z_lower = mean - 3 * std
    z_upper = mean + 3 * std
    z_scores = np.abs((data - mean) / std)
    z_outliers = data[z_scores > 3]
    methods_info.append(('Z-Score', z_outliers, z_lower, z_upper))
    
    # Modified Z-score
    median = data.median()
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    mod_outliers = data[np.abs(modified_z) > 3.5]
    mod_lower = median - 3.5 * mad / 0.6745
    mod_upper = median + 3.5 * mad / 0.6745
    methods_info.append(('Modified Z', mod_outliers, mod_lower, mod_upper))
    
    # Plot each method
    for idx, (name, outliers, lower, upper) in enumerate(methods_info):
        ax = axes[idx]
        
        inliers = data[~data.isin(outliers)]
        
        ax.hist(inliers, bins=50, alpha=0.7, color='blue', label='Inliers', edgecolor='black')
        ax.hist(outliers, bins=20, alpha=0.7, color='red', label='Outliers', edgecolor='black')
        ax.axvline(lower, color='red', linestyle='--', linewidth=2)
        ax.axvline(upper, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{name}: {len(outliers)} outliers')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"📊 Outlier Detection Comparison: {column}")
    print(f"{'='*60}")
    for name, outliers, lower, upper in methods_info:
        print(f"{name:15s}: {len(outliers):5d} outliers ({len(outliers)/len(data)*100:5.2f}%)")


# ==================== SUMMARY & REPORT ====================

def outlier_summary(df, columns=None, methods=['iqr', 'zscore', 'isolation'], verbose=True):
    """
    สรุปผล outliers จากหลายวิธี
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    columns : list, optional
        Columns ที่ต้องการตรวจสอบ
    methods : list
        Methods ที่จะใช้
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    pd.DataFrame
        Summary report
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = detect_outliers_ensemble(df, columns, methods, min_votes=1, verbose=verbose)
    
    # Create summary
    summary_data = []
    for method, outliers_list in results['by_method'].items():
        summary_data.append({
            'Method': method,
            'Outliers': len(outliers_list),
            'Percentage': f"{len(outliers_list)/len(df)*100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n📋 Summary Report:")
    print(summary_df.to_string(index=False))
    
    return summary_df


print("✅ Outliers module loaded successfully!")
print("💡 Available: IQR, Z-score, Isolation Forest, LOF, Elliptic Envelope, Ensemble")