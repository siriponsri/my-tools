# 📖 Models Module - คู่มือการใช้งาน

## 🤖 Model Wrappers ที่มี

### 🔵 Scikit-Learn Models
1. **`SKLearnWrapper`** - Universal wrapper (ใช้กับ sklearn model ใดก็ได้)
2. **`RandomForestWrapper`** - Random Forest พร้อม optimized defaults
3. **`RidgeWrapper`** - Ridge Regression
4. **`LassoWrapper`** - Lasso Regression
5. **`ElasticNetWrapper`** - ElasticNet

### 🟢 Gradient Boosting Models
6. **`LGBWrapper`** - LightGBM (เร็วที่สุด!)
7. **`XGBWrapper`** - XGBoost (แม่นยำ)
8. **`CatBoostWrapper`** - CatBoost (รองรับ categorical โดยตรง)

### 🛠️ Utility Functions
- `quick_model_comparison()` - เปรียบเทียบ regression models
- `quick_classification_comparison()` - เปรียบเทียบ classification models
- `compare_scalers()` - หา scaler ที่ดีที่สุด
- `create_pipeline()` - สร้าง sklearn pipeline

---

## 💡 ตัวอย่างการใช้งาน

### Example 1: Basic Training with CV
```python
from kaggle_utils.models import RandomForestWrapper

# Train with 5-fold CV
rf = RandomForestWrapper(
    n_estimators=200,
    max_depth=10,
    n_splits=5,
    random_state=42,
    task='regression'
)

rf.train(X_train, y_train, X_test=X_test)

# ดูผลลัพธ์
print(f"OOF Predictions: {rf.oof_predictions}")
print(f"Test Predictions: {rf.test_predictions}")
print(f"CV Scores: {rf.scores}")

# Predict new data
new_predictions = rf.predict(X_new)
```

**Output:**
```
============================================================
🚀 Training RandomForestRegressor with 5-Fold CV
============================================================
Fold 1 RMSE: 1234.5678
Fold 2 RMSE: 1235.1234
Fold 3 RMSE: 1236.5432
Fold 4 RMSE: 1234.9876
Fold 5 RMSE: 1235.6789

📊 Overall RMSE: 1235.3802 (±0.6234)
```

### Example 2: LightGBM (แนะนำ!)
```python
from kaggle_utils.models import LGBWrapper

# Custom parameters
lgb = LGBWrapper(
    params={
        'learning_rate': 0.05,
        'num_leaves': 50,
        'max_depth': 7,
        'min_child_samples': 20
    },
    n_splits=5,
    task='regression'
)

# Train
lgb.train(X_train, y_train, X_test=X_test)

# Feature importance
importance_df = lgb.get_feature_importance(X_train.columns, top_n=20)
print(importance_df)
```

### Example 3: XGBoost
```python
from kaggle_utils.models import XGBWrapper

xgb = XGBWrapper(
    params={
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    n_splits=5,
    task='regression'
)

xgb.train(X_train, y_train, X_test=X_test)
```

### Example 4: CatBoost (ดีสำหรับ categorical features)
```python
from kaggle_utils.models import CatBoostWrapper

cat = CatBoostWrapper(
    params={
        'learning_rate': 0.05,
        'depth': 7,
        'iterations': 1000
    },
    n_splits=5,
    task='regression'
)

# CatBoost รองรับ categorical features โดยตรง!
cat_features = ['city', 'category', 'brand']
cat.train(X_train, y_train, X_test=X_test, cat_features=cat_features)
```

### Example 5: Linear Models
```python
from kaggle_utils.models import RidgeWrapper, LassoWrapper, ElasticNetWrapper

# Ridge Regression (L2 regularization)
ridge = RidgeWrapper(alpha=1.0, n_splits=5)
ridge.train(X_train, y_train, X_test=X_test)

# Lasso Regression (L1 regularization - feature selection)
lasso = LassoWrapper(alpha=0.1, n_splits=5)
lasso.train(X_train, y_train, X_test=X_test)

# ElasticNet (L1 + L2)
elastic = ElasticNetWrapper(alpha=1.0, l1_ratio=0.5, n_splits=5)
elastic.train(X_train, y_train, X_test=X_test)
```

### Example 6: Universal SKLearnWrapper
```python
from kaggle_utils.models import SKLearnWrapper
from sklearn.ensemble import GradientBoostingRegressor

# ใช้กับ sklearn model ใดก็ได้!
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

wrapper = SKLearnWrapper(
    model=gbr,
    n_splits=5,
    task='regression'
)

wrapper.train(X_train, y_train, X_test=X_test)
```

### Example 7: Quick Model Comparison
```python
from kaggle_utils.models import quick_model_comparison

# เปรียบเทียบทุก model แบบอัตโนมัติ!
results = quick_model_comparison(
    X_train, y_train,
    cv=5,
    random_state=42
)

print(results)
```

**Output:**
```
🔄 Comparing regression models...
============================================================
Linear Regression   : RMSE = 1450.2345 (±45.2341)
Ridge              : RMSE = 1445.3421 (±44.3214)
Lasso              : RMSE = 1448.1234 (±43.5432)
ElasticNet         : RMSE = 1446.5678 (±44.1234)
Random Forest      : RMSE = 1235.4567 (±35.2345)  ← Best!
Extra Trees        : RMSE = 1238.3214 (±36.1234)
Gradient Boosting  : RMSE = 1240.1234 (±34.5432)
AdaBoost           : RMSE = 1355.6789 (±42.3214)
KNN                : RMSE = 1389.2345 (±48.1234)
============================================================
✅ Best model: Random Forest
   RMSE: 1235.4567
```

### Example 8: Classification
```python
from kaggle_utils.models import RandomForestWrapper

# Classification task
rf_clf = RandomForestWrapper(
    n_estimators=200,
    max_depth=10,
    n_splits=5,
    task='classification'  # เปลี่ยนเป็น classification!
)

rf_clf.train(X_train, y_train, X_test=X_test)
```

### Example 9: Compare Scalers
```python
from kaggle_utils.models import compare_scalers
from sklearn.linear_model import Ridge

# หา scaler ที่ดีที่สุดสำหรับ model นี้
best_scaler, results = compare_scalers(
    X_train, y_train,
    model=Ridge(),
    cv=5
)

print(f"Best scaler: {best_scaler}")
# Output: Best scaler: StandardScaler
```

### Example 10: Create Pipeline
```python
from kaggle_utils.models import create_pipeline
from sklearn.linear_model import Ridge

# สร้าง pipeline พร้อม scaling
pipeline = create_pipeline(
    estimator=Ridge(alpha=1.0),
    scale=True,
    scaler='standard'
)

# Train
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## 🎯 Complete Workflow Example

```python
from kaggle_utils.models import *
from kaggle_utils.preprocessing import *
import pandas as pd

# === 1. LOAD & PREPROCESS ===
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

X = train.drop(columns=['target'])
y = train['target']

# === 2. QUICK COMPARISON ===
print("🔍 Step 1: Quick Model Comparison")
results = quick_model_comparison(X, y, cv=5)
print(results.head())

# === 3. TRAIN TOP MODELS ===
print("\n🚀 Step 2: Training Top Models")

# Random Forest
rf = RandomForestWrapper(n_splits=5)
rf.train(X, y, X_test=test)

# LightGBM
lgb = LGBWrapper(n_splits=5)
lgb.train(X, y, X_test=test)

# XGBoost
xgb = XGBWrapper(n_splits=5)
xgb.train(X, y, X_test=test)

# === 4. ENSEMBLE ===
print("\n🎯 Step 3: Ensemble Predictions")
from kaggle_utils.ensemble import blend_predictions

final_pred = blend_predictions(
    [rf.test_predictions, lgb.test_predictions, xgb.test_predictions],
    weights=[0.3, 0.4, 0.3],
    method='average'
)

# === 5. SUBMIT ===
from kaggle_utils.utils import create_submission
create_submission(test['id'], final_pred, 'submission.csv')
```

---

## 📊 Model Properties & Methods

### Properties (เหมือนกันทุก wrapper)
```python
model.oof_predictions     # Out-of-fold predictions
model.test_predictions    # Test set predictions
model.models             # List of models (1 per fold)
model.scores             # List of scores (1 per fold)
```

### Methods
```python
# Training
model.train(X, y, X_test=None)

# Prediction
model.predict(X_new)

# Feature Importance (สำหรับ tree-based models)
importance_df = model.get_feature_importance(feature_names, top_n=20)
```

---

## 🎨 Customization Examples

### Custom LightGBM Parameters
```python
lgb = LGBWrapper(
    params={
        # Training
        'n_estimators': 2000,
        'learning_rate': 0.03,
        
        # Tree structure
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        
        # Sampling
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        
        # Regularization
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_split_gain': 0.0,
        
        # Other
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'verbose': -1
    },
    n_splits=5,
    task='regression'
)
```

### Custom XGBoost Parameters
```python
xgb = XGBWrapper(
    params={
        # Training
        'n_estimators': 2000,
        'learning_rate': 0.03,
        
        # Tree structure
        'max_depth': 7,
        'min_child_weight': 1,
        'gamma': 0,
        
        # Sampling
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        
        # Regularization
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        
        # Other
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0
    },
    n_splits=5,
    task='regression'
)
```

### Custom Random Forest Parameters
```python
rf = RandomForestWrapper(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    n_splits=5,
    task='regression'
)
```

---

## 🚨 Tips & Best Practices

### 1. 📊 **Cross-Validation is Critical**
```python
# ❌ WRONG - No CV, likely overfitting
model.fit(X_train, y_train)

# ✅ CORRECT - CV built-in
model = RandomForestWrapper(n_splits=5)
model.train(X_train, y_train, X_test)
```

### 2. 🎯 **Use OOF Predictions for Stacking**
```python
# OOF = Out-of-Fold predictions
# ใช้สำหรับ ensemble/stacking
lgb.train(X_train, y_train)
xgb.train(X_train, y_train)

# ใช้ OOF predictions เป็น meta-features
meta_train = pd.DataFrame({
    'lgb_pred': lgb.oof_predictions,
    'xgb_pred': xgb.oof_predictions
})
```

### 3. ⚖️ **Scaling for Linear Models Only**
```python
# Tree-based models (ไม่ต้อง scale)
rf = RandomForestWrapper()  # ไม่ต้อง scale
lgb = LGBWrapper()          # ไม่ต้อง scale

# Linear models (ต้อง scale!)
from kaggle_utils.preprocessing import scale_features
X_scaled, scaler = scale_features(X, method='standard')
ridge = RidgeWrapper()
ridge.train(X_scaled, y)
```

### 4. 🔍 **Model Selection Guide**

| Dataset Size | Features | Recommended Model |
|--------------|----------|-------------------|
| Small (<1K) | Any | Ridge/Lasso |
| Medium (1K-10K) | Numeric | LightGBM, XGBoost |
| Medium (1K-10K) | Many Categorical | CatBoost |
| Large (>10K) | Any | LightGBM (fastest!) |
| High-dimensional | Any | Lasso (feature selection) |

### 5. 📈 **Hyperparameter Tuning Order**
```python
# 1. Start with defaults
lgb = LGBWrapper()
lgb.train(X, y)

# 2. Tune learning_rate first
lgb = LGBWrapper(params={'learning_rate': 0.01})

# 3. Then tune tree structure
lgb = LGBWrapper(params={
    'learning_rate': 0.01,
    'num_leaves': 50,
    'max_depth': 7
})

# 4. Finally tune regularization
lgb = LGBWrapper(params={
    'learning_rate': 0.01,
    'num_leaves': 50,
    'max_depth': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
})
```

### 6. 🎲 **Random State Matters**
```python
# เพื่อความ reproducible
rf = RandomForestWrapper(random_state=42)
lgb = LGBWrapper(random_state=42)
xgb = XGBWrapper(random_state=42)
```

---

## 🔗 Integration with Other Modules

```python
from kaggle_utils import *
from kaggle_utils.models import *
from kaggle_utils.diagnostics import quick_diagnosis, detect_overfitting
from sklearn.model_selection import train_test_split

# 1. Diagnose
report = quick_diagnosis(train, 'target')

# 2. Train
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train, X_test)

# 3. Check overfitting
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
detect_overfitting(lgb.models[0], X_train_split, y_train_split, X_val, y_val)

# 4. Feature importance
lgb.get_feature_importance(X_train.columns, top_n=20)
```

---

## 📌 Quick Reference

| Model Type | Wrapper | When to Use |
|------------|---------|-------------|
| Fast & Accurate | `LGBWrapper` | Medium-Large datasets |
| Very Accurate | `XGBWrapper` | When accuracy is top priority |
| Categorical | `CatBoostWrapper` | Many categorical features |
| Simple & Fast | `RandomForestWrapper` | Quick baseline |
| Regularization | `RidgeWrapper` / `LassoWrapper` | High-dimensional, small data |
| Any sklearn | `SKLearnWrapper` | Custom sklearn models |

---

**💡 Pro Tip:** เริ่มด้วย `quick_model_comparison()` เสมอเพื่อหา baseline ที่ดี!
