# ⚙️ Hyperparameter Tuning Guide

คู่มือการปรับ hyperparameters สำหรับโมเดล Machine Learning

## 📚 Table of Contents
- [Overview](#overview)
- [Available Methods](#available-methods)
- [Grid Search](#grid-search)
- [Random Search](#random-search)
- [Bayesian Optimization (Optuna)](#bayesian-optimization-optuna)
- [Preset Search Spaces](#preset-search-spaces)
- [Examples](#examples)
- [Tips & Best Practices](#tips--best-practices)

---

## Overview

Module นี้ช่วยให้คุณหา hyperparameters ที่ดีที่สุดสำหรับโมเดลของคุณได้อย่างง่ายดาย รองรับ 3 วิธีหลัก:
1. **Grid Search** - ลองทุกค่าใน grid
2. **Random Search** - สุ่มลองตามจำนวนที่กำหนด
3. **Bayesian Optimization** - ใช้ Optuna หาค่าที่ดีที่สุดอัจฉริยะ

---

## Available Methods

### 1. `tune_hyperparameters()`
ฟังก์ชันหลักสำหรับปรับ hyperparameters

```python
from kaggle_utils import tune_hyperparameters

# ใช้ Bayesian Optimization (แนะนำ)
best_params = tune_hyperparameters(
    model=LGBMRegressor(),
    param_space=param_space,
    X=X_train,
    y=y_train,
    method='bayesian',  # 'grid', 'random', 'bayesian'
    cv=5,
    n_trials=50,  # สำหรับ bayesian/random
    scoring='neg_mean_squared_error',
    verbose=True
)
```

### 2. `grid_search_cv()`
Grid Search with Cross-Validation

```python
from kaggle_utils import grid_search_cv

results = grid_search_cv(
    model=RandomForestRegressor(),
    param_grid={
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    X=X_train,
    y=y_train,
    cv=5,
    scoring='r2'
)
```

### 3. `random_search_cv()`
Random Search with Cross-Validation

```python
from kaggle_utils import random_search_cv

results = random_search_cv(
    model=XGBRegressor(),
    param_distributions=param_space,
    X=X_train,
    y=y_train,
    n_iter=50,  # จำนวนครั้งที่จะสุ่มลอง
    cv=5
)
```

### 4. `bayesian_optimization()`
Bayesian Optimization ด้วย Optuna (แนะนำ!)

```python
from kaggle_utils import bayesian_optimization

best_params = bayesian_optimization(
    objective_func=objective,
    n_trials=100,
    direction='minimize',  # 'minimize' หรือ 'maximize'
    study_name='my_optimization',
    verbose=True
)
```

---

## Preset Search Spaces

มี preset search spaces สำหรับโมเดลยอดนิยม:

### LightGBM

```python
from kaggle_utils import suggest_params_lightgbm

# ขนาดของ search space: 'narrow', 'default', 'wide'
params = suggest_params_lightgbm(
    search_space='default',
    task='regression'  # 'regression' หรือ 'classification'
)

# ตัวอย่าง output:
# {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 500, 1000],
#     'num_leaves': [31, 50, 100],
#     'max_depth': [-1, 5, 10],
#     'min_child_samples': [20, 50, 100],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }
```

### XGBoost

```python
from kaggle_utils import suggest_params_xgboost

params = suggest_params_xgboost(
    search_space='default',
    task='regression'
)

# ตัวอย่าง output:
# {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 500, 1000],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }
```

### CatBoost

```python
from kaggle_utils import suggest_params_catboost

params = suggest_params_catboost(
    search_space='default',
    task='classification'
)
```

### Random Forest

```python
from kaggle_utils import suggest_params_random_forest

params = suggest_params_random_forest(
    search_space='wide'
)

# ตัวอย่าง output:
# {
#     'n_estimators': [50, 100, 200, 500, 1000],
#     'max_depth': [None, 5, 10, 15, 20],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 2, 4, 8],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
```

---

## Examples

### Example 1: Quick Tuning ด้วย Grid Search

```python
from kaggle_utils import grid_search_cv, suggest_params_lightgbm
from lightgbm import LGBMRegressor

# เตรียม data
X_train, y_train = ...

# ใช้ preset params
param_grid = suggest_params_lightgbm(search_space='narrow')

# Tune!
results = grid_search_cv(
    model=LGBMRegressor(),
    param_grid=param_grid,
    X=X_train,
    y=y_train,
    cv=5,
    verbose=True
)

print(f"Best params: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")
```

### Example 2: Random Search สำหรับ XGBoost

```python
from kaggle_utils import random_search_cv, suggest_params_xgboost
from xgboost import XGBClassifier

# เตรียม data
X_train, y_train = ...

# ใช้ preset params แบบ wide
param_space = suggest_params_xgboost(
    search_space='wide',
    task='classification'
)

# Random search (เร็วกว่า Grid Search)
results = random_search_cv(
    model=XGBClassifier(),
    param_distributions=param_space,
    X=X_train,
    y=y_train,
    n_iter=50,  # ลอง 50 combinations
    cv=5,
    scoring='roc_auc'
)

# ใช้ best params
best_model = XGBClassifier(**results['best_params'])
best_model.fit(X_train, y_train)
```

### Example 3: Bayesian Optimization (แนะนำ!)

```python
from kaggle_utils import bayesian_optimization
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import optuna

# เตรียม data
X_train, y_train = ...

# สร้าง objective function
def objective(trial):
    # กำหนด search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    # Train และวัดผล
    model = LGBMRegressor(**params, random_state=42, verbose=-1)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    return -scores.mean()  # Optuna จะ minimize

# Run optimization
best_params = bayesian_optimization(
    objective_func=objective,
    n_trials=100,
    direction='minimize',
    study_name='lgbm_optimization',
    verbose=True
)

print(f"Best params: {best_params}")

# Train final model
final_model = LGBMRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
```

### Example 4: ใช้ `tune_hyperparameters()` (ง่ายที่สุด!)

```python
from kaggle_utils import tune_hyperparameters, suggest_params_catboost
from catboost import CatBoostClassifier

# เตรียม data
X_train, y_train = ...

# ใช้ preset params
param_space = suggest_params_catboost(
    search_space='default',
    task='classification'
)

# Tune แบบง่าย!
best_params = tune_hyperparameters(
    model=CatBoostClassifier(verbose=False),
    param_space=param_space,
    X=X_train,
    y=y_train,
    method='bayesian',  # แนะนำ
    cv=5,
    n_trials=50,
    scoring='roc_auc',
    verbose=True
)

# Train final model
model = CatBoostClassifier(**best_params, verbose=False)
model.fit(X_train, y_train)
```

---

## Tips & Best Practices

### 1. เลือก Method อย่างไร?

| Method | ข้อดี | ข้อเสีย | เมื่อไหร่ใช้ |
|--------|-------|---------|--------------|
| **Grid Search** | ลองทุกค่า ครบถ้วน | ช้ามาก (exponential) | Search space เล็ก, ต้องการความแม่นยำสูง |
| **Random Search** | เร็วกว่า Grid | อาจพลาดค่าที่ดี | Search space ใหญ่, เวลาจำกัด |
| **Bayesian (Optuna)** | ฉลาด เร็ว แม่นยำ | ต้อง install optuna | แนะนำ! ใช้ได้เกือบทุกกรณี |

### 2. เริ่มจาก Narrow → Wide

```python
# ครั้งแรก: ใช้ narrow เพื่อหาช่วงคร่าวๆ
params_narrow = suggest_params_lightgbm('narrow')
results_narrow = tune_hyperparameters(..., param_space=params_narrow, n_trials=20)

# จากนั้น: ขยาย search space รอบๆ best params
# แก้ไข search space ตามผลที่ได้
```

### 3. เลือก CV Folds ให้เหมาะสม

```python
# ข้อมูลน้อย (<10K): cv=5 หรือ cv=10
# ข้อมูลปานกลาง (10K-100K): cv=5
# ข้อมูลเยอะ (>100K): cv=3 หรือ cv=5 (เพื่อความเร็ว)
```

### 4. ใช้ Scoring Metric ที่ถูกต้อง

```python
# Regression
scoring='neg_mean_squared_error'  # RMSE
scoring='r2'                       # R²
scoring='neg_mean_absolute_error'  # MAE

# Classification
scoring='roc_auc'        # ROC-AUC
scoring='accuracy'       # Accuracy
scoring='f1'            # F1-Score
scoring='precision'     # Precision
```

### 5. Monitor Overfitting

```python
# ดู gap ระหว่าง train vs validation score
# ถ้า gap ใหญ่ → overfitting

# เพิ่ม regularization:
# - L1: reg_alpha
# - L2: reg_lambda
# - min_child_samples, min_data_in_leaf
```

### 6. ลำดับการ Tune (สำหรับมือใหม่)

```python
# Step 1: Fix learning_rate, tune structure params
params_step1 = {
    'learning_rate': 0.1,  # fix
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [31, 50, 100],
}

# Step 2: Tune sampling params
params_step2 = {
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

# Step 3: Tune regularization
params_step3 = {
    'reg_alpha': [0, 0.01, 0.1, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 1.0],
}

# Step 4: Fine-tune learning_rate + n_estimators
params_step4 = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 2000],
}
```

### 7. ประหยัดเวลา

```python
# 1. ใช้ n_jobs=-1 (parallel)
grid_search_cv(..., n_jobs=-1)

# 2. ใช้ subset ของข้อมูล
X_sample = X_train.sample(n=10000, random_state=42)
y_sample = y_train.loc[X_sample.index]

# 3. ใช้ early_stopping (สำหรับ GBDT)
# จะหยุด train เร็วถ้าไม่ improve
```

### 8. Save Results

```python
import json

# บันทึก best params
with open('best_params.json', 'w') as f:
    json.dump(results['best_params'], f, indent=2)

# โหลดกลับมาใช้
with open('best_params.json', 'r') as f:
    best_params = json.load(f)

model = LGBMRegressor(**best_params)
```

---

## Common Issues

### Issue 1: Optuna ไม่ติดตั้ง
```python
# Error: optuna not available
# Fix:
pip install optuna
```

### Issue 2: ช้ามาก
```python
# ลด cv folds
cv=3  # แทน cv=5

# ลด n_trials
n_trials=20  # แทน n_trials=100

# ใช้ subset ของข้อมูล
X_sample = X_train.sample(10000)
```

### Issue 3: Out of Memory
```python
# ใช้ข้อมูลน้อยลง
X_sample = X_train.sample(n=5000)

# หรือลด cv folds
cv=3
```

---

## สรุป

| ระดับ | วิธีแนะนำ |
|-------|-----------|
| **มือใหม่** | ใช้ `suggest_params_*()` + Grid Search (`grid_search_cv`) |
| **ปานกลาง** | ใช้ Random Search (`random_search_cv`) + preset narrow/default |
| **Advanced** | ใช้ Bayesian Optimization (`bayesian_optimization`) + custom objective |

**Tip สุดท้าย:** อย่าใช้เวลา tune มากเกินไป! บางครั้ง feature engineering ดีกว่า 🚀

---

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
