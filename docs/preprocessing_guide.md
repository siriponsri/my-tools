# 📖 Preprocessing Module - คู่มือการใช้งาน

## 📚 หมวดหมู่ฟังก์ชัน

### 1. 🔍 Data Inspection
- `quick_info()` - ดูข้อมูลสรุป
- `reduce_mem_usage()` - ลด memory usage

### 2. 🛠️ Missing Values
- `handle_missing_values()` - จัดการ missing values
- `create_missing_indicators()` - สร้าง binary features บอกว่า missing หรือไม่

### 3. 🔢 Encoding
- `encode_categorical()` - Encode categorical features
- `target_encode()` - Target encoding with smoothing

### 4. ⚖️ Scaling
- `scale_features()` - Scale numeric features

### 5. 🎯 Feature Engineering
- `create_time_features()` - สร้าง features จากวันที่
- `create_polynomial_features()` - สร้าง polynomial features (x², x*y)
- `create_interaction_features()` - สร้าง interactions (multiply, divide, etc.)
- `create_aggregation_features()` - สร้าง group-by statistics
- `create_ratio_features()` - สร้าง ratio features
- `create_bins()` - แบ่ง continuous เป็น categorical
- `create_text_features()` - สร้าง features จาก text

### 6. ✂️ Feature Selection
- `auto_feature_selection()` - เลือก features ที่ดีที่สุด
- `remove_low_variance_features()` - ลบ features ที่ความแปรปรวนต่ำ

### 7. 🕐 Time Series
- `split_train_test_by_date()` - แบ่ง train/test ตามเวลา

---

## 💡 ตัวอย่างการใช้งาน

### Example 1: Basic Workflow
```python
from kaggle_utils.preprocessing import *

# 1. ดูข้อมูล
quick_info(train, "Training Data")

# 2. ลด memory
train = reduce_mem_usage(train)

# 3. จัดการ missing values (auto)
train = handle_missing_values(train, strategy='auto')

# 4. Encode categorical
train = encode_categorical(train, method='label')

# 5. Scale features
train_scaled, scaler = scale_features(train, method='standard')
```

### Example 2: Missing Values Handling
```python
# วิธีที่ 1: Auto (แนะนำ)
train = handle_missing_values(train, strategy='auto')
# numeric → median, categorical → mode

# วิธีที่ 2: ลบ columns + rows
train = handle_missing_values(train, strategy='drop', threshold=0.5)
# ลบ columns ที่มี missing > 50%, แล้วลบ rows ที่เหลือ

# วิธีที่ 3: สร้าง indicator features (บางครั้ง missing มีความหมาย!)
train = create_missing_indicators(train, columns=['price', 'area'])
# สร้าง 'price_is_missing', 'area_is_missing'
train = handle_missing_values(train, strategy='median')
```

### Example 3: Encoding Categorical Features
```python
# Label Encoding (เหมาะกับ tree-based models)
train = encode_categorical(train, method='label')

# One-Hot Encoding (เหมาะกับ linear models)
train = encode_categorical(train, method='onehot', drop_first=True)

# Frequency Encoding
train = encode_categorical(train, method='frequency')

# Target Encoding (แม่นยำแต่ระวัง overfitting)
train, test = target_encode(
    train, test, 
    cat_cols=['category', 'brand'],
    target_col='price',
    smoothing=10  # เพิ่ม smoothing ถ้ากลัว overfitting
)
```

### Example 4: Time Features (สำคัญมาก!)
```python
# สร้าง 17 time features อัตโนมัติ
train = create_time_features(train, date_col='purchase_date', drop_original=False)

# Features ที่ได้:
# - year, month, day, dayofweek, dayofyear, quarter, week
# - month_sin, month_cos, day_sin, day_cos (cyclical encoding)
# - is_weekend, is_month_start, is_month_end
# - is_quarter_start, is_quarter_end
```

### Example 5: Feature Engineering
```python
# Polynomial Features
X_poly = create_polynomial_features(X, degree=2, interaction_only=False)
# สร้าง: x², y², x*y, etc.

# Interaction Features (manual control)
train = create_interaction_features(
    train,
    col_pairs=[('price', 'area'), ('age', 'rooms')],
    operation='multiply'
)

# Aggregation Features (group statistics)
train = create_aggregation_features(
    train,
    group_cols=['city', 'category'],
    agg_cols=['price', 'area'],
    agg_funcs=['mean', 'std', 'min', 'max']
)
# สร้าง: price_mean_by_city_category, price_std_by_city_category, etc.

# Ratio Features
train = create_ratio_features(
    train,
    numerator_cols=['price'],
    denominator_cols=['area', 'rooms']
)
# สร้าง: price_per_area, price_per_rooms
```

### Example 6: Text Features
```python
# ถ้ามี text column
train = create_text_features(train, text_col='description')

# Features ที่ได้:
# - description_length (ความยาว)
# - description_word_count (จำนวนคำ)
# - description_unique_words (คำไม่ซ้ำ)
# - description_uppercase_count
# - description_digit_count
# - description_special_char_count
```

### Example 7: Feature Selection
```python
# Auto selection
selected_features = auto_feature_selection(
    X_train, y_train,
    k=30,  # เลือก 30 features
    task='auto',  # auto-detect regression/classification
    method='both'  # ใช้ทั้ง F-test และ Mutual Information
)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Remove low variance
train = remove_low_variance_features(train, threshold=0.01)
```

### Example 8: Binning (แบ่งช่วง)
```python
# แบ่ง age เป็น 5 กลุ่ม
train = create_bins(
    train, 
    column='age',
    n_bins=5,
    labels=['very_young', 'young', 'middle', 'old', 'very_old'],
    strategy='quantile'  # แบ่งให้แต่ละกลุ่มมีจำนวนเท่ากัน
)
```

---

## 🎯 Complete Workflow Example

```python
from kaggle_utils.preprocessing import *
import pandas as pd

# === 1. LOAD DATA ===
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target_col = 'price'

# === 2. INSPECT ===
quick_info(train, "Training Data")
quick_info(test, "Test Data")

# === 3. REDUCE MEMORY ===
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# === 4. HANDLE MISSING VALUES ===
# สร้าง missing indicators ก่อน (ถ้าคิดว่า missing มีความหมาย)
train = create_missing_indicators(train, columns=['area', 'rooms'])
test = create_missing_indicators(test, columns=['area', 'rooms'])

# แล้วค่อย fill missing
train = handle_missing_values(train, strategy='auto')
test = handle_missing_values(test, strategy='auto')

# === 5. TIME FEATURES ===
if 'purchase_date' in train.columns:
    train = create_time_features(train, 'purchase_date')
    test = create_time_features(test, 'purchase_date')

# === 6. TEXT FEATURES ===
if 'description' in train.columns:
    train = create_text_features(train, 'description')
    test = create_text_features(test, 'description')

# === 7. FEATURE ENGINEERING ===
# Interaction features
train = create_interaction_features(
    train,
    col_pairs=[('area', 'rooms'), ('age', 'floor')],
    operation='multiply'
)
test = create_interaction_features(
    test,
    col_pairs=[('area', 'rooms'), ('age', 'floor')],
    operation='multiply'
)

# Aggregation features
train = create_aggregation_features(
    train,
    group_cols=['city'],
    agg_cols=['price', 'area'],
    agg_funcs=['mean', 'std']
)
test = create_aggregation_features(
    test,
    group_cols=['city'],
    agg_cols=['price', 'area'],
    agg_funcs=['mean', 'std']
)

# === 8. ENCODING ===
# Label encoding สำหรับ categorical
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != target_col]

train = encode_categorical(train, method='label', columns=cat_cols)
test = encode_categorical(test, method='label', columns=cat_cols)

# หรือใช้ target encoding (ดีกว่าแต่ระวัง overfitting)
# train, test = target_encode(
#     train, test,
#     cat_cols=['category', 'city'],
#     target_col=target_col,
#     smoothing=10
# )

# === 9. FEATURE SELECTION ===
X = train.drop(columns=[target_col])
y = train[target_col]

# Remove low variance
X = remove_low_variance_features(X, threshold=0.01)

# Auto selection
selected_features = auto_feature_selection(X, y, k=50, method='both')
X_selected = X[selected_features]
X_test_selected = test[selected_features]

# === 10. SCALING (optional - ดีสำหรับ linear models) ===
# X_selected, scaler = scale_features(X_selected, method='standard')
# X_test_selected, _ = scale_features(X_test_selected, method='standard')

print(f"\n✅ Preprocessing Complete!")
print(f"Final shape: {X_selected.shape}")
```

---

## 🚨 Tips & Best Practices

### ⚠️ Data Leakage Prevention

**❌ WRONG - Data Leakage:**
```python
# ห้าม! Target encoding ก่อน split
df = target_encode(df, ['category'], 'price')
train, val = train_test_split(df)
```

**✅ CORRECT:**
```python
# Split ก่อน แล้วค่อย encode
train, val = train_test_split(df)
train, val = target_encode(train, val, ['category'], 'price')
```

### 🎯 Feature Engineering Order

**แนะนำลำดับ:**
1. Handle missing values
2. Create time features (ถ้ามี)
3. Create domain-specific features
4. Create interaction/aggregation features
5. Encoding categorical
6. Feature selection
7. Scaling (ถ้าจำเป็น)

### 💡 Target Encoding Tips

```python
# ใช้ smoothing สูงๆ ถ้า:
# - Categories มีจำนวนข้อมูลน้อย
# - กลัว overfitting

train, test = target_encode(
    train, test,
    cat_cols=['rare_category'],
    target_col='price',
    smoothing=100  # สูง = ป้องกัน overfitting
)
```

### 🔄 Cyclical Encoding

**สำคัญมากสำหรับ time features!**
```python
# ❌ WRONG - Model ไม่รู้ว่า month=12 ใกล้กับ month=1
df['month'] = df['date'].dt.month

# ✅ CORRECT - Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
```

### 📊 When to Scale?

**ต้อง Scale:**
- Linear models (Ridge, Lasso, Logistic)
- Neural Networks
- SVM
- K-Nearest Neighbors

**ไม่จำเป็นต้อง Scale:**
- Tree-based models (Random Forest, XGBoost, LightGBM)
- Naive Bayes

---

## 🔗 Integration with Other Modules

```python
from kaggle_utils import *
from kaggle_utils.preprocessing import *
from kaggle_utils.diagnostics import quick_diagnosis

# 1. Preprocess
train = reduce_mem_usage(train)
train = handle_missing_values(train, strategy='auto')

# 2. Diagnose
report = quick_diagnosis(train, target_col='price')

# 3. Feature Engineering (ตามคำแนะนำจาก report)
if 'purchase_date' in train.columns:
    train = create_time_features(train, 'purchase_date')

# 4. Train Model
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train, X_test)
```

---

## 📌 Quick Reference

| Task | Function | Example |
|------|----------|---------|
| ดูข้อมูล | `quick_info()` | `quick_info(train)` |
| ลด memory | `reduce_mem_usage()` | `train = reduce_mem_usage(train)` |
| Fill missing | `handle_missing_values()` | `train = handle_missing_values(train, 'auto')` |
| Encode | `encode_categorical()` | `train = encode_categorical(train, 'label')` |
| Scale | `scale_features()` | `train, scaler = scale_features(train, 'standard')` |
| Time features | `create_time_features()` | `train = create_time_features(train, 'date')` |
| Interactions | `create_interaction_features()` | `train = create_interaction_features(train, [('a','b')])` |
| Aggregations | `create_aggregation_features()` | `train = create_aggregation_features(train, ['city'], ['price'])` |
| Feature selection | `auto_feature_selection()` | `features = auto_feature_selection(X, y, k=30)` |

---

**💡 Pro Tip:** ใช้ `quick_info()` บ่อยๆ หลังแต่ละขั้นตอนเพื่อตรวจสอบว่าข้อมูลเป็นอย่างไร!
