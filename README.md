# 🚀 Kaggle Utils

**ชุดเครื่องมือสำหรับมือใหม่หัดแข่ง Kaggle** 🎯

Universal toolkit for Kaggle competitions - เหมาะสำหรับผู้เริ่มต้นและผู้ที่ต้องการทำ baseline ได้รวดเร็ว

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **สำหรับมือใหม่** - เริ่มต้นแข่ง Kaggle ได้ง่าย ไม่ต้องเขียนโค้ดซ้ำๆ
- 🔍 **Data Diagnostics** - ตรวจสอบคุณภาพข้อมูล ตรวจจับ leakage อัตโนมัติ
- 🤖 **Model Wrappers** - Train models พร้อม CV ในบรรทัดเดียว
- 🎨 **Interactive Viz** - Visualizations แบบ interactive ด้วย Plotly
- ⚡ **Fast & Easy** - ประหยัดเวลา focus ที่ feature engineering และ modeling

## 📦 Installation

### 🚀 Quick Start (สำหรับ Colab/Kaggle) - แนะนำ!

ใช้เพียง **2 บรรทัด** บน Colab/Kaggle:

```python
!wget https://raw.githubusercontent.com/YOUR_USERNAME/kaggle-utils/main/kaggle_utils_single.py
from kaggle_utils_single import *
```

**ข้อดี:**
- ⚡ รวดเร็ว (ไม่ต้อง install)
- 🎯 มี functions หลักๆ ครบ
- 💡 เหมาะสำหรับ quick experiments

📖 **[อ่านคู่มือ Single-File Version →](docs/single_file_guide.md)**

---

### 📦 Full Package Installation (สำหรับ Local/Production)

```bash
# Basic installation (scikit-learn only)
pip install -e .

# Full installation (with LightGBM, XGBoost, CatBoost, Optuna)
pip install -e ".[full]"

# Or install from GitHub (เมื่อ upload แล้ว)
pip install git+https://github.com/yourusername/kaggle-utils.git
```

**ข้อดี:**
- 🎨 Features ครบทั้งหมด (100+ functions)
- 🤖 Model wrappers, Ensemble, Hyperparameter tuning
- 📊 Interactive visualizations
- 🔧 Production-ready

## 🎯 Quick Start

```python
from kaggle_utils import *

# 1. Setup environment (สำหรับ Colab)
setup_colab()
setup_kaggle()

# 2. Load and inspect data
train = load_data('train.csv', show_info=True)
test = load_data('test.csv')

# 3. 🔍 ตรวจสอบข้อมูล (สำหรับมือใหม่!)
report = quick_diagnosis(train, target_col='price', test_df=test)
# จะแนะนำว่าควรทำอะไรต่อ!

# 4. Reduce memory
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# 5. Quick model comparison
results = quick_model_comparison(X_train, y_train, cv=5)

# 6. Train best model with wrapper
lgb = LGBWrapper(n_splits=5, verbose=True)
lgb.train(X_train, y_train, X_test)

# 7. Create submission
create_submission(
    ids=test['Id'],
    predictions=lgb.test_predictions,
    filename='submission.csv',
    id_column='Id',
    target_column='SalePrice'
)
```

## 📚 Modules Overview

### 1. 🔧 Preprocessing (`preprocessing.py`)

**Data Inspection & Cleaning:**
- `quick_info()` - แสดงข้อมูลสรุป missing values, dtypes
- `reduce_mem_usage()` - ลด memory usage 50-75%
- `handle_missing_values()` - จัดการ missing values หลายวิธี

**Feature Engineering:**
- `create_time_features()` - สร้าง features จาก datetime
- `create_polynomial_features()` - Polynomial features
- `create_interaction_features()` - Interaction features
- `create_aggregation_features()` - Group aggregation features
- `target_encode()` - Target encoding with smoothing
- `auto_feature_selection()` - เลือก features อัตโนมัติ

📖 **[อ่านคู่มือเต็ม →](docs/preprocessing_guide.md)**

---

### 2. 🤖 Models (`models.py`)

**Scikit-Learn Wrappers (with built-in CV):**
- `SKLearnWrapper` - Universal wrapper สำหรับ sklearn models
- `RandomForestWrapper` - Random Forest
- `RidgeWrapper`, `LassoWrapper`, `ElasticNetWrapper` - Linear models

**Gradient Boosting Wrappers:**
- `LGBWrapper` - LightGBM (fast & powerful!)
- `XGBWrapper` - XGBoost (stable & accurate)
- `CatBoostWrapper` - CatBoost (handles categorical well)

**Model Comparison:**
- `quick_model_comparison()` - เปรียบเทียบหลาย models พร้อม CV
- `quick_classification_comparison()` - สำหรับ classification
- `compare_scalers()` - เปรียบเทียบ scalers

📖 **[อ่านคู่มือเต็ม →](docs/models_guide.md)**

---

### 3. 🎯 Ensemble (`ensemble.py`)

**Ensemble Methods:**
- `WeightedEnsemble` - Weighted average ensemble
- `StackingEnsemble` - Stacking with meta-learner
- `DynamicEnsemble` - Dynamic weight optimization
- `create_voting_ensemble()` - Voting ensemble (hard/soft)

**Blending:**
- `blend_predictions()` - Average, rank, geometric mean
- `optimize_blend_weights()` - หาน้ำหนักที่ดีที่สุด

📖 **[อ่านคู่มือเต็ม →](docs/ensemble_guide.md)**

---

### 4. 🔍 Outliers (`outliers.py`)

**Detection Methods:**
- `detect_outliers_iqr()` - IQR method (classic)
- `detect_outliers_zscore()` - Z-score method
- `detect_outliers_isolation_forest()` - Isolation Forest
- `detect_outliers_lof()` - Local Outlier Factor
- `detect_outliers_ensemble()` - รวมหลายวิธี

**Handling & Visualization:**
- `handle_outliers()` - Cap, remove, or transform
- `plot_outliers()` - Interactive outlier visualization
- `outlier_summary()` - สรุปรายงาน outliers

📖 **[อ่านคู่มือเต็ม →](docs/outliers_guide.md)**

---

### 5. ⚙️ Hyperparameters (`hyperparams.py`)

**Tuning Methods:**
- `tune_hyperparameters()` - All-in-one tuning function
- `grid_search_cv()` - Grid Search with CV
- `random_search_cv()` - Random Search with CV
- `bayesian_optimization()` - Bayesian Optimization (Optuna)

**Preset Search Spaces:**
- `suggest_params_lightgbm()` - LightGBM params (narrow/default/wide)
- `suggest_params_xgboost()` - XGBoost params
- `suggest_params_catboost()` - CatBoost params
- `suggest_params_random_forest()` - Random Forest params

📖 **[อ่านคู่มือเต็ม →](docs/hyperparams_guide.md)**

---

### 6. 🎨 Visualization (`visualization.py`)

**Interactive Plots (Plotly):**
- `plot_feature_importance()` - Feature importance (single/comparison)
- `plot_distributions()` - Train vs test distributions
- `plot_correlation_heatmap()` - Correlation heatmap
- `plot_learning_curves()` - Learning curves
- `plot_confusion_matrix()` - Confusion matrix
- `plot_roc_curve()` - ROC curve
- `plot_predictions()` - Actual vs predicted

📖 **[อ่านคู่มือเต็ม →](docs/visualization_guide.md)**

---

### 7. 📊 Metrics (`metrics.py`)

**Regression Metrics:**
- `rmse()`, `mae()`, `mape()`, `rmsle()`, `r2_score_custom()`
- `calculate_regression_metrics()` - คำนวณ metrics ทั้งหมด

**Classification Metrics:**
- `calculate_classification_metrics()` - Accuracy, Precision, Recall, F1, ROC-AUC
- `confusion_matrix_metrics()` - Metrics from confusion matrix
- `optimal_threshold()` - หา threshold ที่ดีที่สุด

**Kaggle-Specific:**
- `kaggle_metric()` - ใช้ metric ตามที่ competition กำหนด

📖 **[อ่านคู่มือเต็ม →](docs/metrics_guide.md)**

---

### 8. 🔍 Diagnostics (`diagnostics.py`) - **สำหรับมือใหม่!** ⭐

**Data Quality:**
- `check_data_quality()` - ตรวจสอบคุณภาพข้อมูลโดยรวม
- `detect_leakage()` - ตรวจจับ data leakage
- `check_multicollinearity()` - ตรวจสอบ multicollinearity

**Model Diagnostics:**
- `suggest_models()` - แนะนำ models ที่เหมาะสม
- `detect_overfitting()` - ตรวจจับ overfitting/underfitting
- `plot_learning_curve()` - Learning curve analysis

**All-in-One:**
- `quick_diagnosis()` - 🌟 **วินิจฉัยครบจบในฟังก์ชันเดียว!**

📖 **[อ่านคู่มือเต็ม →](docs/diagnostics_guide.md)**

---

### 9. 🛠️ Utils (`utils.py`)

**Environment Setup:**
- `setup_colab()` - Setup Google Colab
- `setup_kaggle()` - Setup Kaggle API
- `check_environment()` - ตรวจสอบ environment

**Data I/O:**
- `load_data()` - โหลดข้อมูล (CSV, Parquet, Excel, JSON)
- `save_data()` - บันทึกข้อมูล (auto-detect format)
- `download_kaggle_dataset()` - ดาวน์โหลดจาก Kaggle

**Submission:**
- `create_submission()` - สร้างไฟล์ submission

**Timing & Memory:**
- `timer()` - Decorator วัดเวลา
- `Timer` - Context manager วัดเวลา
- `memory_usage()` - ดู memory usage

**Others:**
- `set_seed()` - Set random seed
- `notify()` - ส่ง notification (Colab)

📖 **[อ่านคู่มือเต็ม →](docs/utils_guide.md)**

---

## 💡 Usage Examples

### Example 1: สำหรับมือใหม่ - Complete Pipeline

```python
from kaggle_utils import *
import pandas as pd

# 1. Setup (ถ้าใช้ Colab)
setup_colab()
setup_kaggle()

# 2. Load data
train = load_data('train.csv', show_info=True)
test = load_data('test.csv')

# 3. 🔍 วินิจฉัยข้อมูล (สำคัญ!)
report = quick_diagnosis(train, target_col='SalePrice', test_df=test)
# จะบอกว่า:
# - มี missing values ไหม
# - มี data leakage ไหม
# - มี outliers ไหม
# - ควรใช้ model อะไร

# 4. Preprocess based on recommendations
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# Handle missing values
train = handle_missing_values(train, strategy='auto')
test = handle_missing_values(test, strategy='auto')

# 5. Feature engineering
X_train = train.drop('SalePrice', axis=1)
y_train = train['SalePrice']
X_test = test.copy()

# 6. Quick model comparison
print("🔍 Comparing models...")
results = quick_model_comparison(X_train, y_train, cv=5)

# 7. Train best model
print("\n🚀 Training LightGBM...")
lgb = LGBWrapper(n_splits=5, verbose=True)
lgb.train(X_train, y_train, X_test)

# 8. Create submission
create_submission(
    ids=test['Id'],
    predictions=lgb.test_predictions,
    filename='submission.csv',
    id_column='Id',
    target_column='SalePrice'
)

print("\n✅ Done! Ready to submit to Kaggle!")
```

### Example 2: Ensemble Multiple Models

```python
from kaggle_utils import *

# Train multiple models
models = {}

print("Training Random Forest...")
models['rf'] = RandomForestWrapper(n_estimators=100, n_splits=5)
models['rf'].train(X_train, y_train, X_test)

print("Training LightGBM...")
models['lgb'] = LGBWrapper(n_splits=5)
models['lgb'].train(X_train, y_train, X_test)

print("Training XGBoost...")
models['xgb'] = XGBWrapper(n_splits=5)
models['xgb'].train(X_train, y_train, X_test)

# Get predictions
predictions = [
    models['rf'].test_predictions,
    models['lgb'].test_predictions,
    models['xgb'].test_predictions
]

# Blend with optimal weights
final_pred = blend_predictions(
    predictions,
    method='optimize',  # หาน้ำหนักที่ดีที่สุดอัตโนมัติ
    y_true=y_val  # ใช้ validation set
)

# Create submission
create_submission(test['Id'], final_pred, 'ensemble_submission.csv')
```

### Example 3: Hyperparameter Tuning

```python
from kaggle_utils import *

# Get preset parameter space
params = suggest_params_lightgbm(
    search_space='default',
    task='regression'
)

# Tune with Bayesian Optimization (แนะนำ!)
best_params = tune_hyperparameters(
    model=LGBMRegressor(),
    param_space=params,
    X=X_train,
    y=y_train,
    method='bayesian',
    cv=5,
    n_trials=50,
    verbose=True
)

print(f"Best params: {best_params}")

# Train final model with best params
final_model = LGBWrapper(**best_params, n_splits=5)
final_model.train(X_train, y_train, X_test)
```

### Example 4: Outlier Detection & Handling

```python
from kaggle_utils import *

# Detect outliers (หลายวิธี)
outliers_iqr = detect_outliers_iqr(train, columns=['SalePrice', 'GrLivArea'])
outliers_iso = detect_outliers_isolation_forest(train)

# Visualize
plot_outliers(train, 'SalePrice', method='iqr')
plot_outliers_comparison(train, 'SalePrice')

# Handle outliers
train_clean = handle_outliers(
    train, 
    columns=['SalePrice', 'GrLivArea'],
    method='cap',  # 'cap', 'remove', 'winsorize'
    threshold=1.5
)

print(f"Removed {len(train) - len(train_clean)} outliers")
```

### Example 5: Feature Engineering Pipeline

```python
from kaggle_utils import *

# 1. Time features (ถ้ามี datetime)
train = create_time_features(train, date_col='transaction_date')

# 2. Polynomial features
X_poly = create_polynomial_features(
    train[numeric_cols],
    degree=2,
    include_bias=False
)

# 3. Interaction features
X_interact = create_interaction_features(
    train[['col1', 'col2', 'col3']],
    max_interactions=2
)

# 4. Aggregation features
train = create_aggregation_features(
    train,
    group_col='category',
    agg_cols=['price', 'quantity'],
    agg_funcs=['mean', 'std', 'max', 'min']
)

# 5. Target encoding
train, encoder = target_encode(
    train,
    categorical_cols=['category', 'brand'],
    target_col='price',
    smoothing=10
)

# 6. Auto feature selection
X_selected, selected_features = auto_feature_selection(
    X_train, y_train,
    task='regression',
    k=50  # select top 50 features
)

print(f"Selected features: {selected_features}")
```

---

## 🎓 For Beginners (คำแนะนำสำหรับมือใหม่)

### เริ่มต้นอย่างไร?

1. **เริ่มจาก Diagnostics เสมอ!** 🔍

```python
# ทุก competition เริ่มจากนี้
report = quick_diagnosis(train, target_col='target', test_df=test)
```

ระบบจะบอกว่า:
- ข้อมูลมีปัญหาอะไรบ้าง
- ควรใช้ model อะไร
- มี data leakage ไหม
- ควรทำอะไรต่อ

2. **ใช้ Model Comparison หา baseline** 🤖

```python
# เปรียบเทียบหลาย models ในครั้งเดียว
results = quick_model_comparison(X_train, y_train, cv=5)
```

3. **Train ด้วย Wrappers (มี CV built-in)** ⚡

```python
# Train + CV ในบรรทัดเดียว!
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train, X_test)
```

4. **สร้าง Submission** 📤

```python
create_submission(test_ids, predictions, 'submission.csv')
```

### Workflow แนะนำ

```python
# Step 1: Diagnose
report = quick_diagnosis(train, target_col='target', test_df=test)

# Step 2: Clean data
train = reduce_mem_usage(train)
train = handle_missing_values(train)

# Step 3: Compare models
results = quick_model_comparison(X_train, y_train)

# Step 4: Train best model
model = LGBWrapper(n_splits=5)
model.train(X_train, y_train, X_test)

# Step 5: Submit
create_submission(test_ids, model.test_predictions, 'submission.csv')
```

### Tips สำหรับมือใหม่

✅ **DO:**
- เริ่มจาก simple model ก่อน (Random Forest, LightGBM)
- ใช้ `quick_diagnosis()` ตั้งแต่เริ่ม
- ลด memory ด้วย `reduce_mem_usage()`
- ใช้ CV อย่างน้อย 5 folds
- เช็ค train vs validation gap

❌ **DON'T:**
- อย่าใช้ model ซับซ้อนตั้งแต่แรก
- อย่า optimize hyperparameters ก่อนมี good features
- อย่าลืมเช็ค data leakage
- อย่าทำ feature engineering มากเกินไป ในตอนแรก

---

## 📁 Project Structure

```
kaggle-utils/
├── kaggle_utils/
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning & feature engineering
│   ├── models.py             # Model wrappers with CV
│   ├── ensemble.py           # Ensemble methods
│   ├── outliers.py           # Outlier detection & handling
│   ├── hyperparams.py        # Hyperparameter tuning
│   ├── visualization.py      # Interactive plots
│   ├── metrics.py            # Evaluation metrics
│   ├── diagnostics.py        # Data & model diagnostics
│   └── utils.py              # Utility functions
├── docs/                     # Documentation
│   ├── preprocessing_guide.md
│   ├── models_guide.md
│   ├── ensemble_guide.md
│   ├── outliers_guide.md
│   ├── hyperparams_guide.md
│   ├── metrics_guide.md
│   ├── visualization_guide.md
│   ├── diagnostics_guide.md
│   └── utils_guide.md
├── setup.py
├── README.md
└── LICENSE
```

---

## 🔧 For Google Colab Users

```python
# วิธี 1: Install จาก GitHub (เมื่อ upload แล้ว)
!pip install git+https://github.com/yourusername/kaggle-utils.git

# วิธี 2: Clone และ install
!git clone https://github.com/yourusername/kaggle-utils.git
%cd kaggle-utils
!pip install -e ".[full]"

# วิธี 3: Upload zip file
# 1. Upload kaggle-utils.zip to Colab
# 2. Unzip และ install
!unzip kaggle-utils.zip
%cd kaggle-utils
!pip install -e .

# ใช้งาน
from kaggle_utils import *
setup_colab()  # Setup Colab environment
```

---

## 📖 Documentation

แต่ละ module มีคู่มือแยกโดยละเอียด:

- 📘 [Preprocessing Guide](docs/preprocessing_guide.md) - Data cleaning & feature engineering
- 📙 [Models Guide](docs/models_guide.md) - Model training with CV
- 📕 [Ensemble Guide](docs/ensemble_guide.md) - Ensemble methods
- 📗 [Outliers Guide](docs/outliers_guide.md) - Outlier detection & handling
- 📔 [Hyperparams Guide](docs/hyperparams_guide.md) - Hyperparameter tuning
- 📓 [Metrics Guide](docs/metrics_guide.md) - Evaluation metrics
- 📖 [Visualization Guide](docs/visualization_guide.md) - Interactive plots
- 📒 [Diagnostics Guide](docs/diagnostics_guide.md) - Data & model diagnostics ⭐
- 📑 [Utils Guide](docs/utils_guide.md) - Utility functions

---

## 🚀 Features Highlights

### ⚡ Fast & Easy
- Train models with CV ในบรรทัดเดียว
- Auto feature selection
- One-line submission creation

### 🔍 Beginner-Friendly
- `quick_diagnosis()` - วินิจฉัยข้อมูลครบจบ
- Data leakage detection
- Model recommendations
- Clear error messages

### 🎨 Interactive
- Plotly-based visualizations
- Progress bars ด้วย tqdm
- Real-time feedback

### 📊 Production-Ready
- Memory optimization (50-75% reduction)
- Type hints สำหรับ IDE support
- Comprehensive documentation

---

## 🤝 Contributing

Contributions are welcome! เรายินดีรับ:

- 🐛 Bug reports
- 💡 Feature requests
- 📝 Documentation improvements
- 🔧 Code contributions

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

สร้างขึ้นเพื่อชุมชน Kaggle โดยเฉพาะมือใหม่ที่เริ่มต้นแข่ง Kaggle 🎯

Built with ❤️ for Kaggle beginners and competitions

---

## 📮 Contact & Support

- 🐙 **GitHub**: [yourusername/kaggle-utils](https://github.com/yourusername/kaggle-utils)
- 📧 **Issues**: [Report bugs or request features](https://github.com/yourusername/kaggle-utils/issues)
- 💬 **Discussions**: [Ask questions](https://github.com/yourusername/kaggle-utils/discussions)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

## 📈 Version History

### v1.0.0 (Current)
- ✅ Initial release
- ✅ 9 modules with comprehensive documentation
- ✅ Support for scikit-learn, LightGBM, XGBoost, CatBoost
- ✅ Interactive visualizations with Plotly
- ✅ Diagnostics module for beginners
- ✅ Complete Thai documentation

---

**Happy Kaggling! 🎯🚀**

## 📁 Project Structure

```
kaggle_utils/
├── __init__.py
├── preprocessing.py
├── models.py
├── ensemble.py
├── outliers.py
├── hyperparams.py
├── visualization.py
├── metrics.py
├── diagnostics.py      🆕 Data validation & model diagnostics
└── utils.py
```

## 🔧 For Colab Users

```python
# Option 1: Install from GitHub
!pip install git+https://github.com/yourusername/kaggle-utils.git

# Option 2: Clone and install
!git clone https://github.com/yourusername/kaggle-utils.git
%cd kaggle-utils
!pip install -e .

# Import and use
from kaggle_utils import *
setup_colab()
```

## 🎓 For ML Beginners

Kaggle Utils มี **Diagnostics Module** ที่ออกแบบมาเพื่อมือใหม่โดยเฉพาะ!

**เริ่มต้นทุกโปรเจคด้วย:**
```python
from kaggle_utils.diagnostics import quick_diagnosis

# วินิจฉัยครบจบในฟังก์ชันเดียว!
report = quick_diagnosis(
    train_df=train,
    target_col='target',
    test_df=test
)
```

**ระบบจะตรวจสอบให้อัตโนมัติ:**
- ✅ คุณภาพข้อมูล (missing values, constant features)
- ✅ Data Leakage (features ที่รั่วไหลข้อมูล)
- ✅ Multicollinearity (features ที่ซ้ำซ้อน)
- ✅ แนะนำโมเดลที่เหมาะสม
- ✅ ตรวจจับ Overfitting/Underfitting

**และให้คำแนะนำชัดเจนว่าควรทำอะไรต่อ!** 💡

📖 อ่าน [Diagnostics Guide](docs/diagnostics_guide.md) สำหรับคำแนะนำโดยละเอียด

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 🙏 Acknowledgments

Built for the Kaggle community with ❤️

## 📮 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
