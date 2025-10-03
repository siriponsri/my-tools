"""
Data Preprocessing and Feature Engineering
สำหรับทำความสะอาดข้อมูลและสร้าง features ใหม่
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, 
    mutual_info_regression, mutual_info_classif
)
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA INSPECTION ====================

def quick_info(df, name="DataFrame"):
    """
    แสดงข้อมูลสรุปของ DataFrame อย่างครอบคลุม
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่ต้องการตรวจสอบ
    name : str
        ชื่อของ DataFrame เพื่อแสดงผล
    
    Returns:
    --------
    pd.DataFrame
        แสดง 5 แถวแรกของข้อมูล
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
    
    # Numeric vs Categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    print(f"\n💾 First few rows:")
    return df.head()

def reduce_mem_usage(df, verbose=True):
    """
    ลด memory usage ของ DataFrame โดยแปลง data types
    
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
        print(f'💾 Memory usage: {start_mem:.2f} MB → {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df

# ==================== MISSING VALUES ====================

def handle_missing_values(df, strategy='auto', fill_value=None, threshold=0.5):
    """
    จัดการ missing values ด้วยวิธีต่างๆ
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame ที่มี missing values
    strategy : str
        วิธีการจัดการ: 'auto', 'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill'
    fill_value : any
        ค่าที่ใช้ fill (ถ้าระบุ)
    threshold : float
        ลบ columns ที่มี missing > threshold (เฉพาะ strategy='drop')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ที่จัดการ missing values แล้ว
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        # ลบ columns ที่มี missing มากเกินไป
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"🗑️  Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        # ลบ rows ที่เหลือ
        df_clean = df_clean.dropna()
        print(f"🗑️  Dropped rows with missing values")
        
    elif strategy == 'auto':
        # Auto: numeric → median, categorical → mode
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"✅ Filled missing values: numeric→median, categorical→mode")
        
    elif strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        print(f"✅ Filled missing values with mean")
        
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        print(f"✅ Filled missing values with median")
        
    elif strategy == 'mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        print(f"✅ Filled missing values with mode")
        
    elif strategy == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill')
        print(f"✅ Forward filled missing values")
        
    elif strategy == 'backward_fill':
        df_clean = df_clean.fillna(method='bfill')
        print(f"✅ Backward filled missing values")
        
    elif fill_value is not None:
        df_clean = df_clean.fillna(fill_value)
        print(f"✅ Filled missing values with {fill_value}")
    
    return df_clean

def create_missing_indicators(df, columns=None):
    """
    สร้าง binary features บอกว่า missing หรือไม่
    บางครั้ง missing value ก็เป็นข้อมูลที่มีค่า!
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    columns : list
        columns ที่ต้องการสร้าง indicator (None = ทั้งหมด)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame พร้อม missing indicator columns
    """
    df_new = df.copy()
    cols = columns if columns else df.columns
    
    for col in cols:
        if df[col].isnull().any():
            df_new[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    print(f"✅ Created missing indicators for {len([c for c in cols if df[c].isnull().any()])} columns")
    return df_new

# ==================== ENCODING ====================

def encode_categorical(df, method='label', columns=None, drop_first=False):
    """
    Encode categorical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    method : str
        'label' (LabelEncoder), 'onehot' (OneHotEncoder), 'frequency', 'target' (ต้องใช้ target_encode())
    columns : list
        columns ที่ต้องการ encode (None = ทั้งหมด)
    drop_first : bool
        สำหรับ onehot - drop first category เพื่อหลีกเลี่ยง multicollinearity
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ที่ encode แล้ว
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if method == 'label':
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        print(f"✅ Label encoded {len(columns)} columns")
        
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=drop_first)
        print(f"✅ One-hot encoded {len(columns)} columns → {df_encoded.shape[1]} total features")
        
    elif method == 'frequency':
        for col in columns:
            if col in df_encoded.columns:
                freq = df_encoded[col].value_counts(normalize=True)
                df_encoded[f'{col}_freq'] = df_encoded[col].map(freq)
        print(f"✅ Frequency encoded {len(columns)} columns")
    
    return df_encoded

def target_encode(train_df, test_df, cat_cols, target_col, smoothing=1):
    """
    Target Encoding with smoothing (ลด overfitting)
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    cat_cols : list
        Categorical columns ที่ต้องการ encode
    target_col : str
        Target column name
    smoothing : float
        Smoothing parameter (เพิ่มเพื่อลด overfitting)
    
    Returns:
    --------
    tuple
        (train_df, test_df) ที่ encode แล้ว
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in cat_cols:
        # คำนวณ target mean สำหรับแต่ละ category
        target_mean = train_df.groupby(col)[target_col].mean()
        global_mean = train_df[target_col].mean()
        counts = train_df.groupby(col).size()
        
        # Smoothing: ยิ่งมีข้อมูลน้อย ยิ่งเข้าใกล้ global mean
        smooth_target = (counts * target_mean + smoothing * global_mean) / (counts + smoothing)
        
        # Apply to train and test
        train_df[f'{col}_target_enc'] = train_df[col].map(smooth_target)
        test_df[f'{col}_target_enc'] = test_df[col].map(smooth_target)
        
        # Fill missing with global mean
        train_df[f'{col}_target_enc'].fillna(global_mean, inplace=True)
        test_df[f'{col}_target_enc'].fillna(global_mean, inplace=True)
    
    print(f"✅ Target encoded {len(cat_cols)} columns (smoothing={smoothing})")
    return train_df, test_df

# ==================== SCALING ====================

def scale_features(df, method='standard', columns=None):
    """
    Scale numeric features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    method : str
        'standard' (StandardScaler), 'minmax' (MinMaxScaler), 'robust' (RobustScaler)
    columns : list
        columns ที่ต้องการ scale (None = numeric ทั้งหมด)
    
    Returns:
    --------
    tuple
        (df_scaled, scaler) - DataFrame ที่ scale แล้วและ scaler object
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")
    
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    print(f"✅ Scaled {len(columns)} columns using {method} scaling")
    
    return df_scaled, scaler

# ==================== FEATURE ENGINEERING ====================

def create_time_features(df, date_col, drop_original=False):
    """
    สร้าง features จากคอลัมน์วันที่
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    date_col : str
        ชื่อ column ที่เป็นวันที่
    drop_original : bool
        ลบ column เดิมหรือไม่
    
    Returns:
    --------
    pd.DataFrame
        DataFrame พร้อม time features
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic features
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
    df[f'{date_col}_quarter'] = df[date_col].dt.quarter
    df[f'{date_col}_week'] = df[date_col].dt.isocalendar().week
    
    # Cyclical features (เพื่อให้โมเดลเข้าใจว่า Dec → Jan ต่อเนื่องกัน)
    df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
    df[f'{date_col}_day_sin'] = np.sin(2 * np.pi * df[date_col].dt.day / 31)
    df[f'{date_col}_day_cos'] = np.cos(2 * np.pi * df[date_col].dt.day / 31)
    
    # Boolean features
    df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df[f'{date_col}_is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df[f'{date_col}_is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    if drop_original:
        df = df.drop(columns=[date_col])
    
    print(f"✅ Created 17 time features from {date_col}")
    return df

def create_polynomial_features(X, degree=2, interaction_only=False, include_bias=False):
    """
    สร้าง polynomial features (x^2, x*y, etc.)
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame with numeric features
    degree : int
        Polynomial degree (2 = x^2, x*y)
    interaction_only : bool
        True = เฉพาะ interactions (x*y), ไม่สร้าง x^2
    include_bias : bool
        เพิ่ม bias term (ค่าคงที่ 1)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with polynomial features
    """
    poly = PolynomialFeatures(
        degree=degree, 
        interaction_only=interaction_only, 
        include_bias=include_bias
    )
    
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    print(f"✅ Created polynomial features: {X.shape[1]} → {X_poly_df.shape[1]} features")
    return X_poly_df

def create_interaction_features(df, col_pairs, operation='multiply'):
    """
    สร้าง interaction features จาก feature pairs
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    col_pairs : list of tuples
        [(col1, col2), (col3, col4), ...]
    operation : str
        'multiply', 'divide', 'add', 'subtract'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with interaction features
    """
    df_new = df.copy()
    
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            if operation == 'multiply':
                df_new[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            elif operation == 'divide':
                df_new[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)  # หลีกเลี่ยง div by zero
            elif operation == 'add':
                df_new[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            elif operation == 'subtract':
                df_new[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
    print(f"✅ Created {len(col_pairs)} interaction features ({operation})")
    return df_new

def create_aggregation_features(df, group_cols, agg_cols, agg_funcs=['mean', 'std', 'min', 'max']):
    """
    สร้าง aggregation features (group by และคำนวณ statistics)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    group_cols : list
        Columns ที่ใช้ group by
    agg_cols : list
        Columns ที่ต้องการคำนวณ aggregation
    agg_funcs : list
        Functions ที่ใช้: 'mean', 'std', 'min', 'max', 'sum', 'count', 'median'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with aggregation features
    """
    df_new = df.copy()
    
    for agg_col in agg_cols:
        for agg_func in agg_funcs:
            if agg_func == 'mean':
                df_new[f'{agg_col}_mean_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('mean')
            elif agg_func == 'std':
                df_new[f'{agg_col}_std_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('std')
            elif agg_func == 'min':
                df_new[f'{agg_col}_min_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('min')
            elif agg_func == 'max':
                df_new[f'{agg_col}_max_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('max')
            elif agg_func == 'sum':
                df_new[f'{agg_col}_sum_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('sum')
            elif agg_func == 'count':
                df_new[f'{agg_col}_count_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('count')
            elif agg_func == 'median':
                df_new[f'{agg_col}_median_by_{"_".join(group_cols)}'] = df.groupby(group_cols)[agg_col].transform('median')
    
    n_features = len(agg_cols) * len(agg_funcs)
    print(f"✅ Created {n_features} aggregation features")
    return df_new

def create_ratio_features(df, numerator_cols, denominator_cols):
    """
    สร้าง ratio features (แบ่งกัน)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    numerator_cols : list
        Columns ที่เป็นตัวตั้ง
    denominator_cols : list
        Columns ที่เป็นตัวหาร
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ratio features
    """
    df_new = df.copy()
    
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col != den_col:
                df_new[f'{num_col}_per_{den_col}'] = df[num_col] / (df[den_col] + 1e-5)
    
    n_features = len(numerator_cols) * len(denominator_cols)
    print(f"✅ Created {n_features} ratio features")
    return df_new

# ==================== FEATURE SELECTION ====================

def auto_feature_selection(X, y, k=20, task='auto', method='both'):
    """
    Feature selection อัตโนมัติ
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    k : int
        จำนวน features ที่ต้องการเลือก
    task : str
        'auto', 'regression', 'classification'
    method : str
        'f_test', 'mutual_info', 'both'
    
    Returns:
    --------
    list
        รายชื่อ features ที่ถูกเลือก
    """
    # Detect task type
    if task == 'auto':
        if y.dtype == 'object' or y.nunique() < 20:
            task = 'classification'
        else:
            task = 'regression'
    
    print(f"🎯 Feature Selection (task={task}, k={k})")
    
    # Select methods
    if task == 'regression':
        selector_f = SelectKBest(f_regression, k=min(k, X.shape[1]))
        selector_mi = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    else:
        selector_f = SelectKBest(f_classif, k=min(k, X.shape[1]))
        selector_mi = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    
    if method in ['f_test', 'both']:
        selector_f.fit(X, y)
        scores_f = pd.DataFrame({
            'feature': X.columns,
            'f_score': selector_f.scores_
        }).sort_values('f_score', ascending=False)
        
        print("\n📊 Top Features by F-statistic:")
        print(scores_f.head(min(k, 10)))
    
    if method in ['mutual_info', 'both']:
        selector_mi.fit(X, y)
        scores_mi = pd.DataFrame({
            'feature': X.columns,
            'mi_score': selector_mi.scores_
        }).sort_values('mi_score', ascending=False)
        
        print("\n📊 Top Features by Mutual Information:")
        print(scores_mi.head(min(k, 10)))
    
    # Return top k features
    if method == 'f_test':
        selected_features = scores_f.head(k)['feature'].tolist()
    elif method == 'mutual_info':
        selected_features = scores_mi.head(k)['feature'].tolist()
    else:  # both - combine scores
        scores_f['rank_f'] = scores_f['f_score'].rank(ascending=False)
        scores_mi['rank_mi'] = scores_mi['mi_score'].rank(ascending=False)
        combined = scores_f.merge(scores_mi[['feature', 'rank_mi']], on='feature')
        combined['combined_rank'] = combined['rank_f'] + combined['rank_mi']
        selected_features = combined.nsmallest(k, 'combined_rank')['feature'].tolist()
    
    print(f"\n✅ Selected {len(selected_features)} features")
    return selected_features

def remove_low_variance_features(df, threshold=0.01):
    """
    ลบ features ที่มีความแปรปรวนต่ำ (ค่าเกือบจะเหมือนกันหมด)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    threshold : float
        Variance threshold (0.01 = 1%)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame ที่ลบ low variance features แล้ว
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var()
    low_var_cols = variances[variances < threshold].index.tolist()
    
    if low_var_cols:
        df_clean = df.drop(columns=low_var_cols)
        print(f"🗑️  Removed {len(low_var_cols)} low variance features")
        print(f"   Features: {low_var_cols[:10]}")
        return df_clean
    else:
        print("✅ No low variance features found")
        return df

# ==================== BINNING ====================

def create_bins(df, column, n_bins=5, labels=None, strategy='quantile'):
    """
    แบ่ง continuous variable เป็น bins (categorical)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    column : str
        Column ที่ต้องการ bin
    n_bins : int
        จำนวน bins
    labels : list
        ชื่อของแต่ละ bin
    strategy : str
        'quantile' (equal frequency), 'uniform' (equal width)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame พร้อม binned feature
    """
    df_new = df.copy()
    
    if labels is None:
        labels = [f'{column}_bin_{i}' for i in range(n_bins)]
    
    if strategy == 'quantile':
        df_new[f'{column}_binned'] = pd.qcut(df[column], q=n_bins, labels=labels, duplicates='drop')
    else:  # uniform
        df_new[f'{column}_binned'] = pd.cut(df[column], bins=n_bins, labels=labels)
    
    print(f"✅ Created bins for {column}: {n_bins} bins ({strategy} strategy)")
    return df_new

# ==================== TEXT FEATURES ====================

def create_text_features(df, text_col):
    """
    สร้าง features จาก text column
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    text_col : str
        Text column name
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with text features
    """
    df_new = df.copy()
    
    # Basic features
    df_new[f'{text_col}_length'] = df[text_col].astype(str).apply(len)
    df_new[f'{text_col}_word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df_new[f'{text_col}_unique_words'] = df[text_col].astype(str).apply(lambda x: len(set(x.split())))
    df_new[f'{text_col}_uppercase_count'] = df[text_col].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))
    df_new[f'{text_col}_digit_count'] = df[text_col].astype(str).apply(lambda x: sum(1 for c in x if c.isdigit()))
    df_new[f'{text_col}_special_char_count'] = df[text_col].astype(str).apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()))
    
    print(f"✅ Created 6 text features from {text_col}")
    return df_new

# ==================== UTILITIES ====================

def split_train_test_by_date(df, date_col, test_size=0.2):
    """
    แบ่ง train/test ตามเวลา (สำหรับ time series)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    date_col : str
        Date column
    test_size : float
        สัดส่วนของ test set
    
    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    df_sorted = df.sort_values(date_col)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"✅ Split by date: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df

print("✅ Preprocessing module loaded successfully!")