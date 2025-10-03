# Preprocessing API Reference

## Data Inspection

### `quick_info`

Display comprehensive information about a DataFrame.

```python
kaggle_utils.quick_info(df, name='DataFrame')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | The DataFrame to inspect |
| `name` | `str` | `'DataFrame'` | Display name for the DataFrame |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | Returns the first 5 rows of the DataFrame |

**Examples:**

```python
import pandas as pd
from kaggle_utils import quick_info

# Load data
df = pd.read_csv('train.csv')

# Display information
quick_info(df, name='Training Data')
```

**Output:**
```
============================================================
Training Data Information
============================================================
Shape: 1,460 rows × 81 columns
Memory: 9.23 MB

Missing Values:
                Missing  Percent
PoolQC             1453    99.52
MiscFeature        1406    96.30
Alley              1369    93.77

Data Types:
int64      35
object     43
float64     3

Numeric features: 38
Categorical features: 43
```

---

### `reduce_mem_usage`

Reduce memory usage of a DataFrame by downcasting numeric types.

```python
kaggle_utils.reduce_mem_usage(df, verbose=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | DataFrame to optimize |
| `verbose` | `bool` | `True` | Print memory reduction details |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with optimized data types |

**Algorithm:**

The function performs the following optimizations:
1. **Integer columns**: Downcast to smallest possible type (int8/int16/int32/int64)
2. **Float columns**: Downcast to float32 where possible
3. **Object columns**: No changes (kept as-is)

**Examples:**

```python
from kaggle_utils import reduce_mem_usage

# Reduce memory usage
df_optimized = reduce_mem_usage(df, verbose=True)
```

**Output:**
```
Memory usage decreased from 450.32 MB to 112.58 MB (75.0% reduction)
```

**Memory Savings:**

| Data Type | Original | Optimized | Savings |
|-----------|----------|-----------|---------|
| int64 → int8 | 8 bytes | 1 byte | 87.5% |
| int64 → int16 | 8 bytes | 2 bytes | 75.0% |
| float64 → float32 | 8 bytes | 4 bytes | 50.0% |

**See Also:**

- [`quick_info()`](#quick_info) - Display DataFrame information
- `pd.DataFrame.memory_usage()` - Pandas memory usage method

---

## Missing Value Handling

### `handle_missing_values`

Handle missing values using various strategies.

```python
kaggle_utils.handle_missing_values(
    df,
    strategy='auto',
    fill_value=None,
    categorical_fill='mode',
    numeric_fill='median',
    threshold=0.5
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | DataFrame with missing values |
| `strategy` | `str` | `'auto'` | Strategy to handle missing values |
| `fill_value` | `Any` | `None` | Custom fill value |
| `categorical_fill` | `str` | `'mode'` | Strategy for categorical columns |
| `numeric_fill` | `str` | `'median'` | Strategy for numeric columns |
| `threshold` | `float` | `0.5` | Drop columns with missing > threshold |

**Strategy Options:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `'auto'` | Automatically choose best strategy | General purpose |
| `'drop'` | Drop rows with any missing values | High-quality data needed |
| `'mean'` | Fill with mean value | Numeric, normally distributed |
| `'median'` | Fill with median value | Numeric, with outliers |
| `'mode'` | Fill with most frequent value | Categorical data |
| `'forward_fill'` | Propagate last valid observation | Time series data |
| `'interpolate'` | Linear interpolation | Continuous numeric data |
| `'constant'` | Fill with specified value | Domain-specific default |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with missing values handled |

**Examples:**

**Example 1: Auto strategy**
```python
from kaggle_utils import handle_missing_values

# Automatically handle missing values
df_clean = handle_missing_values(df, strategy='auto')
```

**Example 2: Custom strategies per type**
```python
# Median for numeric, mode for categorical
df_clean = handle_missing_values(
    df,
    numeric_fill='median',
    categorical_fill='mode'
)
```

**Example 3: Drop high-missing columns**
```python
# Drop columns with >50% missing
df_clean = handle_missing_values(
    df,
    strategy='auto',
    threshold=0.5
)
```

**Example 4: Time series forward fill**
```python
# For time series data
df_clean = handle_missing_values(
    df,
    strategy='forward_fill'
)
```

**Behavior by Column Type:**

| Column Type | Auto Strategy | Rationale |
|-------------|---------------|-----------|
| Numeric (no outliers) | Mean | Central tendency |
| Numeric (with outliers) | Median | Robust to outliers |
| Categorical | Mode | Most common value |
| Datetime | Forward fill | Temporal continuity |

**Notes:**

- Columns with >threshold missing are dropped
- String 'missing' is treated as actual missing value
- Preserves column data types
- Creates backup before modification (optional)

**See Also:**

- [`create_missing_indicators()`](#create_missing_indicators) - Create binary indicators for missing values
- `pd.DataFrame.fillna()` - Pandas fill missing method
- `sklearn.impute.SimpleImputer` - Scikit-learn imputer

---

## Feature Engineering

### `create_time_features`

Extract features from datetime columns.

```python
kaggle_utils.create_time_features(
    df,
    date_col,
    features='all',
    drop_original=False,
    prefix=None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | Input DataFrame |
| `date_col` | `str` or `list` | *required* | Datetime column(s) to process |
| `features` | `str` or `list` | `'all'` | Features to extract |
| `drop_original` | `bool` | `False` | Drop original datetime column |
| `prefix` | `str` | `None` | Prefix for new columns |

**Available Features:**

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `year` | Year | int | 1970-2100+ |
| `month` | Month | int | 1-12 |
| `day` | Day of month | int | 1-31 |
| `dayofweek` | Day of week | int | 0-6 (Mon-Sun) |
| `dayofyear` | Day of year | int | 1-366 |
| `week` | Week of year | int | 1-53 |
| `quarter` | Quarter | int | 1-4 |
| `hour` | Hour | int | 0-23 |
| `minute` | Minute | int | 0-59 |
| `is_weekend` | Weekend flag | bool | 0-1 |
| `is_month_start` | Month start flag | bool | 0-1 |
| `is_month_end` | Month end flag | bool | 0-1 |
| `is_quarter_start` | Quarter start flag | bool | 0-1 |
| `is_quarter_end` | Quarter end flag | bool | 0-1 |
| `is_year_start` | Year start flag | bool | 0-1 |
| `is_year_end` | Year end flag | bool | 0-1 |
| `elapsed` | Seconds since epoch | int | Timestamp |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with additional time features |

**Examples:**

**Example 1: Extract all features**
```python
from kaggle_utils import create_time_features

# Create all time features
df = create_time_features(df, date_col='transaction_date')
```

**Example 2: Extract specific features**
```python
# Extract only year, month, day
df = create_time_features(
    df,
    date_col='date',
    features=['year', 'month', 'day']
)
```

**Example 3: Multiple date columns**
```python
# Process multiple datetime columns
df = create_time_features(
    df,
    date_col=['start_date', 'end_date'],
    features=['year', 'month', 'dayofweek']
)
```

**Example 4: Custom prefix**
```python
# Add prefix to avoid name conflicts
df = create_time_features(
    df,
    date_col='purchase_date',
    prefix='purchase'
)
# Creates: purchase_year, purchase_month, etc.
```

**Feature Importance:**

Common feature importance in time-based competitions:

| Feature | Importance | Use Cases |
|---------|------------|-----------|
| `month` | High | Seasonal patterns, retail |
| `dayofweek` | High | Weekly patterns, traffic |
| `hour` | Medium | Intraday patterns, usage |
| `is_weekend` | Medium | Behavior differences |
| `quarter` | Medium | Business cycles |
| `year` | Low | Long-term trends |

**Tips:**

- Always check for timezone issues before extraction
- Consider cyclical encoding for periodic features (month, hour)
- `is_weekend` often more useful than raw `dayofweek`
- Remove `elapsed` if not needed to save memory

**Cyclical Encoding Example:**

For features like month or hour, consider cyclical encoding:

```python
import numpy as np

# Month as cyclical features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**See Also:**

- `pd.to_datetime()` - Convert to datetime
- `pd.DatetimeIndex` - Datetime index operations

---

### `create_polynomial_features`

Create polynomial and interaction features.

```python
kaggle_utils.create_polynomial_features(
    df,
    columns=None,
    degree=2,
    interaction_only=False,
    include_bias=False,
    prefix='poly'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | Input DataFrame |
| `columns` | `list` | `None` | Columns to create features from. If None, use all numeric |
| `degree` | `int` | `2` | Maximum degree of polynomial features |
| `interaction_only` | `bool` | `False` | Only create interaction terms, no powers |
| `include_bias` | `bool` | `False` | Include bias column (all ones) |
| `prefix` | `str` | `'poly'` | Prefix for new column names |

**Returns:**

| Type | Description |
|------|-------------|
| `pd.DataFrame` | DataFrame with polynomial features added |

**Feature Generation:**

For `degree=2` and features `[a, b]`:

| Feature | Formula | Example (a=2, b=3) |
|---------|---------|-------------------|
| `a` | a | 2 |
| `b` | b | 3 |
| `a²` | a × a | 4 |
| `a × b` | a × b | 6 |
| `b²` | b × b | 9 |

**Examples:**

**Example 1: Second-degree polynomials**
```python
from kaggle_utils import create_polynomial_features

# Create degree-2 polynomial features
df_poly = create_polynomial_features(
    df[['feature1', 'feature2']],
    degree=2
)
```

**Example 2: Interaction terms only**
```python
# Only create interactions, no powers
df_interact = create_polynomial_features(
    df,
    columns=['age', 'income', 'credit_score'],
    interaction_only=True
)
# Creates: age×income, age×credit_score, income×credit_score
```

**Example 3: Higher degree**
```python
# Third-degree polynomials
df_poly3 = create_polynomial_features(
    df,
    columns=['x', 'y'],
    degree=3
)
# Creates: x, y, x², xy, y², x³, x²y, xy², y³
```

**Complexity Analysis:**

Number of features created:

| Input Features | Degree | Output Features | Formula |
|----------------|--------|-----------------|---------|
| 2 | 2 | 5 | (n+d)!/(n!×d!) - 1 |
| 3 | 2 | 9 | |
| 5 | 2 | 20 | |
| 10 | 2 | 65 | |
| 5 | 3 | 55 | |

**Warning:**

**Exponential Growth**: Features grow rapidly with degree and number of columns!

- For 10 features with degree=3: 285 new features
- For 20 features with degree=2: 230 new features

**Best Practices:**

1. **Start small**: Begin with degree=2
2. **Select features**: Only use most important features
3. **Feature selection**: Use feature selection after creation
4. **Watch memory**: Monitor memory usage with many features
5. **Regularization**: Use L1/L2 regularization with models

**Use Cases:**

| Scenario | Recommendation |
|----------|----------------|
| Linear regression | degree=2, all features |
| Tree-based models | Usually not needed |
| Neural networks | degree=2, selected features |
| Small datasets | degree=2-3 carefully |
| Large datasets | interaction_only=True |

**See Also:**

- [`create_interaction_features()`](#create_interaction_features) - Manual interaction creation
- `sklearn.preprocessing.PolynomialFeatures` - Underlying implementation

---

### `target_encode`

Encode categorical variables using target statistics.

```python
kaggle_utils.target_encode(
    df,
    categorical_cols,
    target_col,
    smoothing=10,
    min_samples_leaf=1,
    noise_level=0,
    cv_folds=5
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *required* | Input DataFrame |
| `categorical_cols` | `list` | *required* | Categorical columns to encode |
| `target_col` | `str` | *required* | Target variable column |
| `smoothing` | `float` | `10` | Smoothing factor (higher = more regularization) |
| `min_samples_leaf` | `int` | `1` | Minimum samples to consider a category |
| `noise_level` | `float` | `0` | Standard deviation of Gaussian noise |
| `cv_folds` | `int` | `5` | Cross-validation folds to prevent overfitting |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple` | `(df_encoded, encoders)` where encoders is a dict of fitted encoders |

**Algorithm:**

Target encoding with smoothing:

```
encoded_value = (n * category_mean + smoothing * global_mean) / (n + smoothing)
```

Where:
- `n` = number of samples in category
- `category_mean` = target mean for that category
- `global_mean` = overall target mean
- `smoothing` = regularization parameter

**Examples:**

**Example 1: Basic target encoding**
```python
from kaggle_utils import target_encode

# Encode categorical features
df_encoded, encoders = target_encode(
    df,
    categorical_cols=['city', 'category', 'brand'],
    target_col='price'
)
```

**Example 2: High regularization**
```python
# More smoothing for rare categories
df_encoded, encoders = target_encode(
    df,
    categorical_cols=['rare_category'],
    target_col='target',
    smoothing=50  # Higher smoothing
)
```

**Example 3: With noise (prevent overfitting)**
```python
# Add noise for regularization
df_encoded, encoders = target_encode(
    df,
    categorical_cols=['category'],
    target_col='sales',
    noise_level=0.01
)
```

**Example 4: Use encoders on test set**
```python
# Fit on train
train_encoded, encoders = target_encode(
    train,
    categorical_cols=['category'],
    target_col='target'
)

# Apply to test (without target leakage)
for col, encoder in encoders.items():
    test[col] = test[col].map(encoder)
    test[col].fillna(encoder['_global_mean'], inplace=True)
```

**Smoothing Effect:**

| Category | Count | Mean | smoothing=0 | smoothing=10 | smoothing=100 |
|----------|-------|------|-------------|--------------|---------------|
| Common | 1000 | 50 | 50.0 | 50.0 | 49.5 |
| Rare | 5 | 80 | 80.0 | 60.0 | 42.0 |
| Global | - | 40 | - | - | - |

**Advantages:**

- Captures target relationship directly
- Works well with high cardinality
- Single feature per category
- Improves linear model performance

**Disadvantages:**

- Risk of target leakage
- Requires careful CV strategy
- May not work well with small data
- Sensitive to outliers in target

**Best Practices:**

1. **Always use CV**: Prevent target leakage
2. **Tune smoothing**: Based on category frequency
3. **Check for leakage**: Compare train vs CV scores
4. **Handle unseen categories**: Use global mean
5. **Monitor overfitting**: Track train/val gap

**Comparison with Other Encodings:**

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| Target Encoding | High performance | Leakage risk |
| One-Hot | No leakage | High dimensionality |
| Label Encoding | Simple | Implies ordering |
| Count Encoding | Simple | Loses target info |

**See Also:**

- [`encode_categorical()`](#encode_categorical) - Other encoding methods
- `category_encoders.TargetEncoder` - Alternative implementation

---

## Complete Function List

### Data Inspection
- [`quick_info()`](#quick_info)
- [`reduce_mem_usage()`](#reduce_mem_usage)

### Missing Values
- [`handle_missing_values()`](#handle_missing_values)
- `create_missing_indicators()` - Create binary indicators

### Encoding
- `encode_categorical()` - Label, OneHot, Ordinal encoding
- [`target_encode()`](#target_encode)

### Scaling
- `scale_features()` - Standard, MinMax, Robust scaling

### Feature Engineering
- [`create_time_features()`](#create_time_features)
- [`create_polynomial_features()`](#create_polynomial_features)
- `create_interaction_features()` - Manual interactions
- `create_aggregation_features()` - Group-by aggregations
- `create_ratio_features()` - Ratio between features
- `create_bins()` - Binning continuous variables
- `create_text_features()` - Text feature extraction

### Feature Selection
- `auto_feature_selection()` - Automatic selection
- `remove_low_variance_features()` - Remove constants

### Data Splitting
- `split_train_test_by_date()` - Time-based splitting
