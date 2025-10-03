# 📦 Single-File Version Guide

## 🎯 Quick Start for Colab/Kaggle Notebooks

ใช้เพียง 2 บรรทัด!

```python
!wget https://raw.githubusercontent.com/siriponsri/my-tools/main/kaggle_utils_single.py
from kaggle_utils_single import *
```

---

## ✨ Features

Single-file version มี functions หลักๆ ที่ใช้บ่อยที่สุด:

### 📊 Data Operations
- `quick_info()` - แสดงข้อมูลสรุป
- `reduce_mem_usage()` - ลด memory 50-75%
- `load_data()` - โหลดข้อมูล (CSV, Parquet, Excel, JSON)
- `save_data()` - บันทึกข้อมูล (auto-detect format)

### 🔍 Diagnostics
- `quick_diagnosis()` - วินิจฉัยข้อมูลครบจบ
- `check_data_quality()` - ตรวจสอบคุณภาพข้อมูล
- `detect_leakage()` - ตรวจจับ data leakage

### 🤖 Models
- `quick_model_comparison()` - เปรียบเทียบหลาย models

### 🛠️ Utils
- `setup_colab()` - Setup Colab environment
- `setup_kaggle()` - Setup Kaggle API
- `create_submission()` - สร้างไฟล์ submission
- `set_seed()` - Set random seed

---

## 💡 Complete Example

```python
# 1. Download and import
!wget https://raw.githubusercontent.com/YOUR_USERNAME/kaggle-utils/main/kaggle_utils_single.py
from kaggle_utils_single import *

# 2. Setup (if using Colab)
setup_colab()
setup_kaggle()

# 3. Load data
train = load_data('train.csv', show_info=True)
test = load_data('test.csv')

# 4. Diagnose (สำคัญ!)
report = quick_diagnosis(train, target_col='SalePrice', test_df=test)

# 5. Reduce memory
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# 6. Prepare data
X_train = train.drop('SalePrice', axis=1)
y_train = train['SalePrice']
X_test = test.copy()

# 7. Compare models
results = quick_model_comparison(X_train, y_train, cv=5)

# 8. Train your model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predict
predictions = model.predict(X_test)

# 10. Create submission
create_submission(
    ids=test['Id'],
    predictions=predictions,
    filename='submission.csv',
    id_column='Id',
    target_column='SalePrice'
)

print("✅ Done! Ready to submit!")
```

---

## 🆚 Single-File vs Full Package

| Feature | Single-File | Full Package |
|---------|-------------|--------------|
| **Installation** | `!wget ...` (2 sec) | `pip install ...` (1-2 min) |
| **Size** | ~25 KB | Full package |
| **Functions** | Essential only | Complete (100+ functions) |
| **Best for** | Quick notebooks | Production/Local dev |
| **Updates** | Manual re-download | `pip install -U` |
| **Dependencies** | sklearn, pandas, numpy | All optional (LightGBM, XGBoost, etc.) |

---

## 🎓 When to Use Which?

### Use Single-File when:
- ✅ Working on Colab/Kaggle notebooks
- ✅ Quick experiments
- ✅ Don't want to install packages
- ✅ Need basic functionality only
- ✅ Want fast setup

### Use Full Package when:
- ✅ Local development
- ✅ Production code
- ✅ Need advanced features (ensemble, hyperparameter tuning)
- ✅ Need LightGBM/XGBoost/CatBoost wrappers
- ✅ Building reusable pipelines

---

## 📚 Available Functions

### Data Operations
```python
# Load data
train = load_data('train.csv', show_info=True)

# Quick info
quick_info(train, "Training Data")

# Reduce memory
train = reduce_mem_usage(train, verbose=True)

# Save data
save_data(train, 'processed.parquet')  # Auto-detect format
```

### Diagnostics
```python
# Complete diagnosis
report = quick_diagnosis(train, target_col='target', test_df=test)

# Data quality check
quality = check_data_quality(train, target_col='target')

# Leakage detection
suspicious = detect_leakage(train, 'target', test)
```

### Models
```python
# Compare models
results = quick_model_comparison(
    X_train, y_train,
    cv=5,
    task='auto',  # 'regression' or 'classification'
    verbose=True
)
```

### Utils
```python
# Setup Colab
setup_colab()

# Setup Kaggle
setup_kaggle()

# Set seed
set_seed(42)

# Create submission
create_submission(
    ids=test_ids,
    predictions=predictions,
    filename='submission.csv'
)
```

---

## 🔄 Updating

Re-download ไฟล์ใหม่:

```python
!rm kaggle_utils_single.py  # ลบไฟล์เก่า
!wget https://raw.githubusercontent.com/YOUR_USERNAME/kaggle-utils/main/kaggle_utils_single.py
```

---

## 🚀 Pro Tips

### 1. Check Version
```python
from kaggle_utils_single import __version__
print(f"Version: {__version__}")
```

### 2. See Example
```python
from kaggle_utils_single import example_usage
example_usage()
```

### 3. Import Specific Functions
```python
from kaggle_utils_single import (
    quick_info,
    reduce_mem_usage,
    quick_diagnosis,
    create_submission
)
```

### 4. Use with Full Package
```python
# ใช้ single-file สำหรับ quick functions
from kaggle_utils_single import quick_diagnosis, reduce_mem_usage

# ใช้ full package สำหรับ advanced features
from kaggle_utils import LGBWrapper, StackingEnsemble
```

---

## 📝 Notes

- Single-file version มี functions หลักๆ เท่านั้น
- ไม่มี advanced features เช่น:
  - Model wrappers (LGBWrapper, XGBWrapper)
  - Ensemble methods (StackingEnsemble, WeightedEnsemble)
  - Hyperparameter tuning (Optuna integration)
  - Advanced outlier detection
  - Interactive visualizations

- สำหรับ features เหล่านี้ ใช้ full package:
```bash
pip install git+https://github.com/YOUR_USERNAME/kaggle-utils.git
```

---

## 🐛 Troubleshooting

### Issue: wget not found (Windows)
```python
# Use curl instead
!curl -O https://raw.githubusercontent.com/YOUR_USERNAME/kaggle-utils/main/kaggle_utils_single.py
```

### Issue: Import error
```python
# Make sure file is in same directory
import os
print(os.listdir())  # Check if kaggle_utils_single.py exists
```

### Issue: Module conflicts
```python
# If you have full package installed
import sys
sys.path.insert(0, '.')  # Prioritize current directory
from kaggle_utils_single import *
```

---

## 🎉 Quick Links

- 📖 [Full Documentation](../README.md)
- 🐙 [GitHub Repository](https://github.com/YOUR_USERNAME/kaggle-utils)
- 📚 [Full Guides](.)

---

**Happy Kaggling! 🚀**
