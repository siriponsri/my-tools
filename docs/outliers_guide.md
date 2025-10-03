# 📖 Outliers Module - คู่มือการใช้งาน

## 🔍 Outlier Detection Methods ที่มี

### 1. **detect_outliers_iqr()** - IQR Method (แนะนำ!)
วิธีมาตรฐาน ใช้ Interquartile Range

### 2. **detect_outliers_zscore()** - Z-Score Method
ใช้ค่าเบี่ยงเบนมาตรฐาน

### 3. **detect_outliers_isolation_forest()** - Isolation Forest
Machine learning-based, ดีสำหรับ multivariate outliers

### 4. **detect_outliers_lof()** - Local Outlier Factor
ดีสำหรับ local density-based outliers

### 5. **detect_outliers_elliptic()** - Elliptic Envelope
สำหรับ Gaussian distributed data

### 6. **detect_outliers_ensemble()** - Ensemble Method ⭐
รวมหลายวิธีเข้าด้วยกัน (แม่นที่สุด!)

## 🛠️ Handling Methods

### 1. **handle_outliers()** - จัดการ outliers
- `cap` - Winsorizing (แนะนำ!)
- `remove` - ลบออก
- `log` - Log transformation
- `sqrt` - Square root transformation
- `clip` - Clipping

## 📊 Visualization

### 1. **plot_outliers()** - แสดง outliers
### 2. **plot_outliers_comparison()** - เปรียบเทียบวิธี
### 3. **outlier_summary()** - สรุปผล

---

## 💡 ตัวอย่างการใช้งาน

### Example 1: Basic Detection with IQR
```python
from kaggle_utils.outliers import detect_outliers_iqr, plot_outliers

# ตรวจจับ outliers
outliers = detect_outliers_iqr(
    train, 
    columns=['price', 'area', 'rooms'],
    threshold=1.5,  # 1.5 = standard, 3.0 = extreme only
    verbose=True
)

print(f"Found {len(outliers)} outliers")

# Visualize
plot_outliers(train, 'price', method='iqr')
```

**Output:**
```
🔍 Detecting outliers using IQR method (threshold=1.5)
============================================================
Checking columns: 100%|████████████████| 3/3

📊 Results:
   Total outliers found: 156
   Affected rows: 3.12%

📋 Details by column:
   price: 89 outliers
   area: 52 outliers
   rooms: 15 outliers
```

### Example 2: Z-Score Method
```python
from kaggle_utils.outliers import detect_outliers_zscore

# ตรวจจับด้วย Z-score
outliers = detect_outliers_zscore(
    train,
    columns=['price', 'area'],
    threshold=3,  # 3 = standard, 2 = more sensitive
    verbose=True
)
```

### Example 3: Isolation Forest (Advanced)
```python
from kaggle_utils.outliers import detect_outliers_isolation_forest

# ตรวจจับ multivariate outliers
outliers = detect_outliers_isolation_forest(
    train,
    columns=['price', 'area', 'rooms', 'age'],
    contamination=0.1,  # คาดว่ามี 10% outliers
    random_state=42,
    verbose=True
)
```

**Output:**
```
🔍 Detecting outliers using Isolation Forest
============================================================
Contamination: 0.1
Features: 4

🌲 Training Isolation Forest...

📊 Results:
   Total outliers found: 500
   Affected rows: 10.00%
```

### Example 4: Ensemble Detection (แนะนำ! ⭐)
```python
from kaggle_utils.outliers import detect_outliers_ensemble

# ใช้หลายวิธีร่วมกัน
result = detect_outliers_ensemble(
    train,
    columns=['price', 'area', 'rooms'],
    methods=['iqr', 'zscore', 'isolation'],
    min_votes=2,  # ต้องตรวจพบจากอย่างน้อย 2 methods
    verbose=True
)

outliers = result['outliers']
votes = result['votes']

print(f"Ensemble found {len(outliers)} outliers")
```

**Output:**
```
============================================================
🎯 Ensemble Outlier Detection
============================================================
Methods: ['iqr', 'zscore', 'isolation']
Min votes required: 2
Columns: 3

Running methods: 100%|████████████████| 3/3

============================================================
📊 Ensemble Results:
============================================================
   iqr            :   156 outliers
   zscore         :   142 outliers
   isolation      :   500 outliers
   ────────────────────────────────────────
   Ensemble       :    98 outliers (≥2 votes)
```

### Example 5: Handle Outliers (Cap/Winsorize)
```python
from kaggle_utils.outliers import handle_outliers

# Cap outliers (แนะนำ!)
train_clean = handle_outliers(
    train,
    columns=['price', 'area'],
    method='cap',  # cap ที่ bounds
    iqr_threshold=1.5,
    verbose=True
)

# Shape เท่าเดิม แต่ outliers ถูก cap แล้ว
print(f"Original: {train.shape}")
print(f"Cleaned: {train_clean.shape}")
```

**Output:**
```
============================================================
🔧 Handling Outliers
============================================================
Method: cap
Columns: 2
Processing: 100%|████████████████| 2/2
✅ Outliers capped using IQR method

📊 Final shape: (5000, 20)
```

### Example 6: Remove Outliers
```python
# Remove outliers (ระวัง! จะลด rows)
train_clean = handle_outliers(
    train,
    columns=['price', 'area'],
    method='remove',
    iqr_threshold=1.5,
    verbose=True
)
```

**Output:**
```
============================================================
🔧 Handling Outliers
============================================================
Method: remove
Columns: 2
✅ Removed 156 rows (3.12%)

📊 Final shape: (4844, 20)
```

### Example 7: Transform with Log
```python
# Log transformation (ดีสำหรับ skewed data)
train_transformed = handle_outliers(
    train,
    columns=['price', 'area'],
    method='log',
    verbose=True
)
```

**Output:**
```
============================================================
🔧 Handling Outliers
============================================================
Method: log
Columns: 2
Transforming: 100%|████████████████| 2/2
✅ Applied log transformation

📊 Final shape: (5000, 20)
```

### Example 8: Visualization
```python
from kaggle_utils.outliers import plot_outliers, plot_outliers_comparison

# Plot single column
plot_outliers(train, 'price', method='iqr', threshold=1.5)

# Compare methods
plot_outliers_comparison(train, 'price')
```

### Example 9: Summary Report
```python
from kaggle_utils.outliers import outlier_summary

# สรุปผลจากหลายวิธี
summary = outlier_summary(
    train,
    columns=['price', 'area', 'rooms'],
    methods=['iqr', 'zscore', 'isolation', 'lof'],
    verbose=True
)

print(summary)
```

**Output:**
```
         Method  Outliers Percentage
            iqr       156      3.12%
         zscore       142      2.84%
      isolation       500     10.00%
            lof       234      4.68%
```

---

## 🎯 Complete Workflow Examples

### Workflow 1: Basic Outlier Handling
```python
from kaggle_utils.outliers import (
    detect_outliers_iqr, 
    plot_outliers, 
    handle_outliers
)

# 1. Detect
outliers = detect_outliers_iqr(train, ['price', 'area'])

# 2. Visualize
plot_outliers(train, 'price')

# 3. Handle (cap)
train_clean = handle_outliers(train, ['price', 'area'], method='cap')

# 4. Continue with modeling
from kaggle_utils.models import LGBWrapper
lgb = LGBWrapper()
lgb.train(train_clean[features], train_clean['target'])
```

### Workflow 2: Advanced Ensemble Detection
```python
from kaggle_utils.outliers import detect_outliers_ensemble, handle_outliers

# 1. Ensemble detection (แม่นที่สุด)
result = detect_outliers_ensemble(
    train,
    columns=['price', 'area', 'rooms', 'age'],
    methods=['iqr', 'zscore', 'isolation', 'lof'],
    min_votes=3,  # ต้อง 3 methods เห็นด้วย
    verbose=True
)

# 2. Get high-confidence outliers
high_conf_outliers = result['outliers']

# 3. Remove only high-confidence outliers
train_clean = train.drop(high_conf_outliers).reset_index(drop=True)

# 4. Cap the rest
train_clean = handle_outliers(train_clean, ['price', 'area'], method='cap')
```

### Workflow 3: Separate Treatment
```python
# แยกการจัดการตาม severity

# 1. Detect extreme outliers (ลบทิ้ง)
extreme = detect_outliers_iqr(train, ['price'], threshold=3.0)
train_clean = train.drop(extreme).reset_index(drop=True)

# 2. Detect moderate outliers (cap)
moderate = detect_outliers_iqr(train_clean, ['price'], threshold=1.5)
# ไม่ลบ แต่ cap แทน
train_clean = handle_outliers(train_clean, ['price'], method='cap', iqr_threshold=1.5)
```

---

## 🎨 Method Selection Guide

| Method | Best For | Speed | Multivariate |
|--------|----------|-------|--------------|
| **IQR** | General use ⭐ | ⚡⚡⚡ | ❌ |
| **Z-Score** | Gaussian data | ⚡⚡⚡ | ❌ |
| **Isolation Forest** | Complex data | ⚡⚡ | ✅ |
| **LOF** | Local patterns | ⚡ | ✅ |
| **Elliptic** | Gaussian multi | ⚡⚡ | ✅ |
| **Ensemble** | Best accuracy ⭐⭐ | ⚡ | ✅ |

### เมื่อไหร่ใช้อะไร?

**IQR (แนะนำเริ่มต้น):**
```python
# ใช้เมื่อ:
# - ไม่รู้จะใช้อะไร → เริ่มจากนี้!
# - ต้องการความเร็ว
# - Data ไม่ซับซ้อน
outliers = detect_outliers_iqr(train, columns=['price'])
```

**Z-Score:**
```python
# ใช้เมื่อ:
# - Data เป็น Gaussian distribution
# - ต้องการ statistical approach
outliers = detect_outliers_zscore(train, columns=['price'])
```

**Isolation Forest:**
```python
# ใช้เมื่อ:
# - หลาย features (multivariate)
# - Outliers ซับซ้อน
# - Data ไม่เป็น Gaussian
outliers = detect_outliers_isolation_forest(
    train, 
    columns=['price', 'area', 'rooms', 'age']
)
```

**Ensemble (แนะนำสำหรับ production):**
```python
# ใช้เมื่อ:
# - ต้องการความแม่นยำสูง
# - มีเวลาคำนวณ
# - Data สำคัญมาก
result = detect_outliers_ensemble(
    train,
    methods=['iqr', 'zscore', 'isolation'],
    min_votes=2
)
```

---

## 🛠️ Handling Method Selection

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Cap/Winsorize** ⭐ | Keep all data | May distort | Most cases |
| **Remove** | Clean data | Lose data | Extreme outliers only |
| **Log** | Fix skewness | Changes scale | Right-skewed data |
| **Sqrt** | Mild transformation | Limited effect | Mild skewness |

### เมื่อไหร่ใช้อะไร?

**Cap (แนะนำ! ⭐):**
```python
# ใช้เมื่อ:
# - ไม่อยากเสียข้อมูล
# - Outliers ไม่รุนแรงมาก
# - Default choice!
train_clean = handle_outliers(train, ['price'], method='cap')
```

**Remove:**
```python
# ใช้เมื่อ:
# - Outliers รุนแรงมาก
# - มีข้อมูลเยอะ (>10K rows)
# - แน่ใจว่าเป็น errors จริงๆ
train_clean = handle_outliers(train, ['price'], method='remove')
```

**Log Transform:**
```python
# ใช้เมื่อ:
# - Data skewed มาก
# - Price, income, population data
# - ต้องการทำให้เป็น Gaussian
train_clean = handle_outliers(train, ['price'], method='log')
```

---

## 🚨 Tips & Best Practices

### 1. ⚠️ Understand Before Remove
```python
# ❌ DON'T - ลบทันทีโดยไม่ดู
train_clean = handle_outliers(train, columns, method='remove')

# ✅ DO - ดูก่อนว่า outliers คืออะไร
outliers_idx = detect_outliers_iqr(train, ['price'])
print(train.loc[outliers_idx, ['price', 'area']])  # ดูข้อมูล
# แล้วค่อยตัดสินใจว่าจะจัดการยังไง
```

### 2. 🎯 Different Thresholds for Different Features
```python
# Features ต่างกันควรใช้ threshold ต่างกัน
# Price - sensitive
train = handle_outliers(train, ['price'], method='cap', iqr_threshold=1.5)

# Area - less sensitive
train = handle_outliers(train, ['area'], method='cap', iqr_threshold=2.0)
```

### 3. 📊 Visualize First
```python
# ดู distribution ก่อนจัดการ
plot_outliers(train, 'price')

# จะช่วยให้รู้ว่าควรใช้วิธีไหน
```

### 4. 🔄 Check After Handling
```python
# ก่อนจัดการ
plot_outliers(train, 'price')

# จัดการ
train_clean = handle_outliers(train, ['price'], method='cap')

# หลังจัดการ - ตรวจสอบอีกครั้ง
plot_outliers(train_clean, 'price')
```

### 5. 💾 Keep Original Data
```python
# เก็บ original ไว้ก่อน
train_original = train.copy()

# แล้วค่อยจัดการ
train_clean = handle_outliers(train, ['price'], method='cap')

# ถ้าไม่ชอบผล กลับไปใช้ original ได้
```

### 6. 🎲 Ensemble for Important Features
```python
# สำหรับ features สำคัญ ใช้ ensemble
result = detect_outliers_ensemble(
    train,
    columns=['price'],  # feature หลัก
    methods=['iqr', 'zscore', 'isolation'],
    min_votes=2
)
```

---

## 📈 Expected Results

### Before Outlier Handling:
```
price: min=1000, max=10000000, std=500000
      ↑ มี outliers สูงมาก!
```

### After Cap/Winsorize:
```
price: min=1000, max=50000, std=8000
      ↑ ดูปกติขึ้น แต่ยังเก็บข้อมูลไว้
```

### Impact on Model:
```
Before: RMSE = 5000
After:  RMSE = 3500  ← ดีขึ้น 30%!
```

---

## 🔗 Integration Example

```python
from kaggle_utils import *
from kaggle_utils.preprocessing import quick_info
from kaggle_utils.outliers import (
    detect_outliers_ensemble,
    plot_outliers,
    handle_outliers
)
from kaggle_utils.models import LGBWrapper

# 1. Load & Inspect
train = pd.read_csv('train.csv')
quick_info(train)

# 2. Detect outliers
result = detect_outliers_ensemble(
    train,
    columns=['price', 'area'],
    methods=['iqr', 'zscore', 'isolation'],
    min_votes=2
)

# 3. Visualize
plot_outliers(train, 'price')

# 4. Handle
train_clean = handle_outliers(
    train,
    columns=['price', 'area'],
    method='cap',
    iqr_threshold=1.5
)

# 5. Train model
lgb = LGBWrapper(n_splits=5)
lgb.train(train_clean[features], train_clean['target'])
```

---

**💡 Pro Tip:** เริ่มด้วย IQR method ก่อนเสมอ! ถ้าต้องการความแม่นยำมากขึ้นค่อยใช้ Ensemble!
