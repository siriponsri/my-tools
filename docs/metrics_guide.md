# Metrics Guide

คู่มือการใช้งาน Evaluation Metrics สำหรับ Kaggle Competitions

## Table of Contents

- [Overview](#overview)
- [Regression Metrics](#regression-metrics)
- [Classification Metrics](#classification-metrics)
- [Kaggle-Specific Metrics](#kaggle-specific-metrics)
- [Custom Metrics](#custom-metrics)
- [Examples](#examples)
- [Tips & Best Practices](#tips--best-practices)

---

## Overview

Module นี้มี metrics สำหรับวัดประสิทธิภาพของโมเดล ทั้ง regression และ classification พร้อมคำอธิบายที่เข้าใจง่าย

### Quick Start

```python
from kaggle_utils import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    rmse, mae, rmsle
)

# Regression
metrics = calculate_regression_metrics(y_true, y_pred)
print(metrics)

# Classification
metrics = calculate_classification_metrics(y_true, y_pred)
print(metrics)
```

---

## Regression Metrics

### 1. RMSE (Root Mean Squared Error)

**ใช้เมื่อไหร่**: เป็น metric ที่นิยมที่สุด penalize errors ใหญ่มากกว่า errors เล็ก

```python
from kaggle_utils import rmse

score = rmse(y_true, y_pred)
print(f"RMSE: {score:.4f}")
```

**การตีความ**:
- ยิ่งน้อยยิ่งดี (minimum = 0)
- มีหน่วยเดียวกับ target variable
- Sensitive กับ outliers

### 2. MAE (Mean Absolute Error)

**ใช้เมื่อไหร่**: เมื่อไม่ต้องการ penalize outliers มาก

```python
from kaggle_utils import mae

score = mae(y_true, y_pred)
print(f"MAE: {score:.4f}")
```

**การตีความ**:
- ค่าเฉลี่ยของ absolute errors
- น้อยกว่า RMSE เสมอ
- ไม่ sensitive กับ outliers เท่า RMSE

### 3. MAPE (Mean Absolute Percentage Error)

**ใช้เมื่อไหร่**: เมื่อต้องการวัดเป็น percentage

```python
from kaggle_utils import mape

score = mape(y_true, y_pred)
print(f"MAPE: {score:.2f}%")
```

**ข้อควรระวัง**:
- ไม่สามารถใช้ได้ถ้า y_true มีค่า 0
- Biased ต่อ predictions ที่ต่ำกว่าจริง

### 4. RMSLE (Root Mean Squared Log Error)

**ใช้เมื่อไหร่**: เมื่อ target มี range กว้างมาก หรือต้องการ penalize underestimation

```python
from kaggle_utils import rmsle

score = rmsle(y_true, y_pred)
print(f"RMSLE: {score:.4f}")
```

**ข้อดี**:
- ไม่ sensitive กับ scale
- Penalize underestimation มากกว่า overestimation
- นิยมใน Kaggle (เช่น House Prices competition)

**ข้อควรระวัง**:
- y_true และ y_pred ต้อง >= 0

### 5. R² Score (Coefficient of Determination)

**ใช้เมื่อไหร่**: เมื่อต้องการทราบว่าโมเดลอธิบาย variance ได้เท่าไหร่

```python
from kaggle_utils import r2_score_custom

score = r2_score_custom(y_true, y_pred)
print(f"R²: {score:.4f}")
```

**การตีความ**:
- 1.0 = perfect prediction
- 0.0 = โมเดลไม่ดีกว่าการใช้ mean
- < 0 = แย่กว่าการใช้ mean

### Calculate All Regression Metrics

```python
from kaggle_utils import calculate_regression_metrics

metrics = calculate_regression_metrics(y_true, y_pred, verbose=True)

# Output:
# {
#     'rmse': 12.34,
#     'mae': 9.56,
#     'mape': 5.67,
#     'r2': 0.89,
#     'rmsle': 0.23  # ถ้า y >= 0
# }
```

---

## Classification Metrics

### 1. Accuracy

**ใช้เมื่อไหร่**: เมื่อ classes มีสัดส่วนใกล้เคียงกัน

```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")
```

**ข้อควรระวัง**:
- ไม่เหมาะกับ imbalanced data
- ไม่บอกว่า error มาจาก class ไหน

### 2. Precision, Recall, F1-Score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

**การตีความ**:
- **Precision**: จากที่ predict เป็น positive มี positive จริงเท่าไหร่
- **Recall**: จาก positive ทั้งหมด เรา catch ได้เท่าไหร่
- **F1**: Harmonic mean ของ precision และ recall

**เลือกใช้อย่างไร**:
- Cost of False Positive สูง → เน้น Precision
- Cost of False Negative สูง → เน้น Recall
- สมดุล → ใช้ F1

### 3. ROC-AUC (Area Under ROC Curve)

**ใช้เมื่อไหร่**: Binary classification, วัดความสามารถในการแยก classes

```python
from sklearn.metrics import roc_auc_score

# ต้องใช้ probability predictions
auc = roc_auc_score(y_true, y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")
```

**การตีความ**:
- 1.0 = perfect classifier
- 0.5 = random guess
- < 0.5 = worse than random

**ข้อดี**:
- ไม่สนใจ threshold
- เหมาะกับ imbalanced data

### 4. Log Loss

**ใช้เมื่อไหร่**: เมื่อต้องการ penalize confident wrong predictions

```python
from sklearn.metrics import log_loss

loss = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {loss:.4f}")
```

**การตีความ**:
- ยิ่งน้อยยิ่งดี (minimum = 0)
- Penalize wrong predictions ที่มั่นใจมาก

### 5. Confusion Matrix Metrics

```python
from kaggle_utils import confusion_matrix_metrics

metrics = confusion_matrix_metrics(y_true, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
```

### Calculate All Classification Metrics

```python
from kaggle_utils import calculate_classification_metrics

# Binary classification
metrics = calculate_classification_metrics(
    y_true, 
    y_pred,
    y_pred_proba=y_pred_proba,  # optional
    verbose=True
)

# Output:
# {
#     'accuracy': 0.85,
#     'precision': 0.83,
#     'recall': 0.87,
#     'f1': 0.85,
#     'roc_auc': 0.91,  # ถ้ามี y_pred_proba
#     'log_loss': 0.32   # ถ้ามี y_pred_proba
# }
```

---

## Kaggle-Specific Metrics

### Using `kaggle_metric()`

ฟังก์ชันนี้ช่วยให้คุณใช้ metric ตามที่ Kaggle competition กำหนด

```python
from kaggle_utils import kaggle_metric

# RMSE
score = kaggle_metric(y_true, y_pred, metric='rmse')

# RMSLE
score = kaggle_metric(y_true, y_pred, metric='rmsle')

# MAE
score = kaggle_metric(y_true, y_pred, metric='mae')

# ROC-AUC
score = kaggle_metric(y_true, y_pred_proba, metric='roc_auc')

# Log Loss
score = kaggle_metric(y_true, y_pred_proba, metric='log_loss')
```

### Common Kaggle Metrics

| Competition Type | Metric | Function |
|-----------------|--------|----------|
| House Prices | RMSLE | `rmsle()` |
| Titanic | Accuracy | `accuracy_score()` |
| Credit Default | ROC-AUC | `roc_auc_score()` |
| Sales Forecasting | RMSE | `rmse()` |

---

## Custom Metrics

### สร้าง Custom Metric เอง

```python
import numpy as np

def custom_metric(y_true, y_pred):
    """
    Custom metric ของคุณ
    """
    # ตัวอย่าง: Weighted RMSE
    weights = np.where(y_true > threshold, 2.0, 1.0)
    mse = np.average((y_true - y_pred) ** 2, weights=weights)
    return np.sqrt(mse)

score = custom_metric(y_true, y_pred)
```

### ใช้กับ Cross-Validation

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

# สร้าง scorer
custom_scorer = make_scorer(
    custom_metric,
    greater_is_better=False  # ถ้ายิ่งน้อยยิ่งดี
)

# ใช้ใน CV
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring=custom_scorer
)
```

---

## Examples

### Example 1: Regression Competition

```python
from kaggle_utils import (
    calculate_regression_metrics,
    rmse, rmsle
)
import numpy as np

# Train model
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# Evaluate
print("=== Training Set ===")
train_metrics = calculate_regression_metrics(y_train, y_pred_train)

print("\n=== Validation Set ===")
val_metrics = calculate_regression_metrics(y_val, y_pred_val)

# Check for overfitting
train_rmse = rmse(y_train, y_pred_train)
val_rmse = rmse(y_val, y_pred_val)

print(f"\nTrain RMSE: {train_rmse:.4f}")
print(f"Val RMSE: {val_rmse:.4f}")
print(f"Gap: {val_rmse - train_rmse:.4f}")

if val_rmse > train_rmse * 1.2:
    print("⚠️  Possible overfitting!")
```

### Example 2: Classification Competition

```python
from kaggle_utils import (
    calculate_classification_metrics,
    optimal_threshold
)

# Train model
model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Find optimal threshold
best_threshold = optimal_threshold(
    y_val, 
    y_pred_proba,
    metric='f1'  # optimize F1-score
)

print(f"Optimal threshold: {best_threshold:.4f}")

# Use optimal threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)

# Evaluate
metrics = calculate_classification_metrics(
    y_val,
    y_pred,
    y_pred_proba=y_pred_proba
)

print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

### Example 3: Custom Competition Metric

```python
# สมมติ competition ใช้ metric: Mean Log Quadratic Error
def mean_log_quadratic_error(y_true, y_pred):
    """Custom Kaggle metric"""
    return np.mean(np.log1p((y_true - y_pred) ** 2))

# สร้าง scorer
from sklearn.metrics import make_scorer
mlqe_scorer = make_scorer(
    mean_log_quadratic_error,
    greater_is_better=False
)

# ใช้ใน model comparison
from kaggle_utils import quick_model_comparison

results = quick_model_comparison(
    X_train, y_train,
    cv=5,
    scoring=mlqe_scorer
)
```

---

## Tips & Best Practices

### 1. เลือก Metric ให้เหมาะกับ Problem

**Regression**:
- Target มี outliers → ใช้ MAE แทน RMSE
- Target มี range กว้าง → ใช้ RMSLE
- ต้องการ penalize errors ใหญ่ → ใช้ RMSE
- ต้องการ % error → ใช้ MAPE (ระวัง divide by zero)

**Classification**:
- Balanced classes → Accuracy, F1
- Imbalanced classes → ROC-AUC, Precision-Recall
- Cost of errors ไม่เท่ากัน → Custom metric
- Need probabilities → Log Loss

### 2. Monitor Multiple Metrics

```python
# อย่าดู metric เดียว!
metrics = {
    'rmse': rmse(y_true, y_pred),
    'mae': mae(y_true, y_pred),
    'r2': r2_score(y_true, y_pred),
}

# ดู trade-offs
print(pd.DataFrame([metrics]))
```

### 3. Check Train vs Validation Gap

```python
# Rule of thumb:
# - Gap < 10% → OK
# - Gap 10-20% → เริ่มมี overfitting
# - Gap > 20% → overfitting แน่นอน

gap_percent = (val_score - train_score) / train_score * 100
print(f"Gap: {gap_percent:.1f}%")
```

### 4. Use Stratified CV for Classification

```python
from sklearn.model_selection import StratifiedKFold

# แทนที่จะใช้ KFold ธรรมดา
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    # Train และ evaluate
    pass
```

### 5. Threshold Optimization

```python
from kaggle_utils import optimal_threshold

# หา threshold ที่ optimize F1
best_threshold = optimal_threshold(
    y_true,
    y_pred_proba,
    metric='f1'
)

# หรือ maximize precision โดยต้องมี recall >= 0.8
best_threshold = optimal_threshold(
    y_true,
    y_pred_proba,
    metric='precision',
    min_recall=0.8
)
```

### 6. Report Metrics เป็นมืออาชีพ

```python
from kaggle_utils import classification_report_custom

# แสดง report พร้อมคำอธิบาย
report = classification_report_custom(y_true, y_pred)
print(report)
```

---

## Common Mistakes

### 1. ใช้ Accuracy กับ Imbalanced Data

```python
# ❌ Wrong
# Dataset: 95% class 0, 5% class 1
# Model ที่ predict ทุกอันเป็น 0 → Accuracy = 95%!

# ✅ Correct
# ใช้ ROC-AUC, F1-Score, Precision-Recall แทน
```

### 2. ไม่ใช้ Probabilities

```python
# ❌ Wrong
y_pred = model.predict(X)  # hard predictions
auc = roc_auc_score(y_true, y_pred)  # ERROR!

# ✅ Correct
y_pred_proba = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y_true, y_pred_proba)
```

### 3. RMSLE กับค่าลบ

```python
# ❌ Wrong
y_pred = model.predict(X)  # อาจมีค่าลบ
score = rmsle(y_true, y_pred)  # ERROR!

# ✅ Correct
y_pred = np.maximum(0, y_pred)  # clip ค่าลบ
score = rmsle(y_true, y_pred)
```

### 4. ไม่ Check for Overfitting

```python
# ✅ Always compare train vs validation
train_score = rmse(y_train, y_pred_train)
val_score = rmse(y_val, y_pred_val)

print(f"Train: {train_score:.4f}")
print(f"Val: {val_score:.4f}")
print(f"Gap: {val_score - train_score:.4f}")
```

---

## สรุป Metrics แต่ละประเภท

### Regression

| Metric | Best Value | Use When | Sensitive to Outliers |
|--------|------------|----------|----------------------|
| RMSE | 0 | ต้องการ penalize errors ใหญ่ | ✅ Yes |
| MAE | 0 | ไม่ต้องการ penalize outliers | ❌ No |
| MAPE | 0 | ต้องการ % error | ❌ No |
| RMSLE | 0 | Target มี range กว้าง | ❌ No |
| R² | 1 | ต้องการดู explained variance | ✅ Yes |

### Classification

| Metric | Best Value | Use When | Need Probabilities |
|--------|------------|----------|-------------------|
| Accuracy | 1 | Balanced classes | ❌ No |
| Precision | 1 | Cost of FP สูง | ❌ No |
| Recall | 1 | Cost of FN สูง | ❌ No |
| F1-Score | 1 | Balance precision/recall | ❌ No |
| ROC-AUC | 1 | Imbalanced data | ✅ Yes |
| Log Loss | 0 | Need calibrated probabilities | ✅ Yes |

---

## References

- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Kaggle Evaluation Metrics](https://www.kaggle.com/c/about/evaluation)
- [Understanding AUC-ROC](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
