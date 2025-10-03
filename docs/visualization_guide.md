# 📖 Visualization Module - คู่มือการใช้งาน

## 📊 Plotting Functions ที่มี

### 1. **Feature Importance** (2 functions)
- `plot_feature_importance()` - Plot feature importance
- `plot_feature_importance_comparison()` - เปรียบเทียบหลาย models

### 2. **Distribution Plots** (2 functions)
- `plot_distributions()` - Train vs Test distributions
- `plot_target_distribution()` - Target variable distribution

### 3. **Correlation** (2 functions)
- `plot_correlation_heatmap()` - Correlation heatmap
- `plot_correlation_with_target()` - Correlation กับ target

### 4. **Learning Curves** (1 function)
- `plot_learning_curves()` - วิเคราะห์ overfitting/underfitting

### 5. **Classification Metrics** (3 functions)
- `plot_confusion_matrix()` - Confusion matrix
- `plot_roc_curve()` - ROC curve
- `plot_precision_recall_curve()` - Precision-Recall curve

### 6. **Regression Metrics** (1 function)
- `plot_predictions()` - Predictions vs Actual

### 7. **Missing Values** (1 function)
- `plot_missing_values()` - Visualize missing values

---

## 💡 ตัวอย่างการใช้งาน

### Example 1: Feature Importance
```python
from kaggle_utils.visualization import plot_feature_importance
from kaggle_utils.models import LGBWrapper

# Train model
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train)

# Plot feature importance
plot_feature_importance(
    lgb.models[0],
    X_train.columns,
    top_n=20,
    title='LightGBM Feature Importance'
)
```

**Output:**
```
📊 Creating feature importance plot...
```
![Feature Importance Plot]

### Example 2: Compare Multiple Models
```python
from kaggle_utils.visualization import plot_feature_importance_comparison

# Train multiple models
lgb.train(X_train, y_train)
xgb.train(X_train, y_train)
rf.train(X_train, y_train)

# Compare importance
plot_feature_importance_comparison(
    models=[lgb.models[0], xgb.models[0], rf.models[0]],
    feature_names=X_train.columns,
    model_names=['LightGBM', 'XGBoost', 'RandomForest'],
    top_n=15
)
```

**Output:**
```
📊 Creating feature importance comparison...
```
![Comparison Plot]

### Example 3: Distribution Comparison
```python
from kaggle_utils.visualization import plot_distributions

# Compare train vs test distributions
plot_distributions(
    train, 
    test,
    columns=['price', 'area', 'rooms', 'age'],
    ncols=2
)
```

**Output:**
```
📊 Plotting 4 distributions...
Creating plots: 100%|████████████████| 4/4
```
![Distribution Plots]

### Example 4: Target Distribution
```python
from kaggle_utils.visualization import plot_target_distribution

# For regression
plot_target_distribution(y_train, task='regression')

# For classification
plot_target_distribution(y_train, task='classification')
```

### Example 5: Correlation Heatmap
```python
from kaggle_utils.visualization import plot_correlation_heatmap

# Full heatmap
plot_correlation_heatmap(train, figsize=(14, 12))

# Top 20 features only
plot_correlation_heatmap(train, top_n=20)
```

**Output:**
```
📊 Computing pearson correlation...
```
![Correlation Heatmap]

### Example 6: Correlation with Target
```python
from kaggle_utils.visualization import plot_correlation_with_target

# Show top features correlated with target
plot_correlation_with_target(
    train,
    target_col='price',
    top_n=20
)
```

**Output:**
```
📊 Computing correlation with target...
```
![Target Correlation]

### Example 7: Learning Curves
```python
from kaggle_utils.visualization import plot_learning_curves
from sklearn.ensemble import RandomForestRegressor

# Plot learning curves
plot_learning_curves(
    RandomForestRegressor(n_estimators=100),
    X_train,
    y_train,
    cv=5
)
```

**Output:**
```
📈 Generating learning curves...
   This may take a while...

📊 Analysis:
   Final Training Score: 1234.5678
   Final Validation Score: 1345.6789
   Gap: 111.1111 (9.0%)
   ⚠️  Slight overfitting
```
![Learning Curves]

### Example 8: Confusion Matrix
```python
from kaggle_utils.visualization import plot_confusion_matrix

# For binary classification
plot_confusion_matrix(
    y_val,
    y_pred,
    labels=['Class 0', 'Class 1'],
    normalize=True  # Show percentages
)
```

### Example 9: ROC Curve
```python
from kaggle_utils.visualization import plot_roc_curve

# Plot ROC curve
plot_roc_curve(y_val, y_pred_proba)
```

**Output:**
![ROC Curve with AUC = 0.876]

### Example 10: Precision-Recall Curve
```python
from kaggle_utils.visualization import plot_precision_recall_curve

# Plot PR curve
plot_precision_recall_curve(y_val, y_pred_proba)
```

### Example 11: Predictions vs Actual
```python
from kaggle_utils.visualization import plot_predictions

# For regression - scatter plot + residuals
plot_predictions(y_val, y_pred)
```

**Output:**
![Predictions Plot with Residuals]

### Example 12: Missing Values Visualization
```python
from kaggle_utils.visualization import plot_missing_values

# Visualize missing values
plot_missing_values(train)
```

**Output:**
```
📊 Summary:
   Total columns with missing: 5
   Highest missing: col_A (156 / 3.12%)
```
![Missing Values Plots]

---

## 🎯 Complete Workflow Examples

### Workflow 1: EDA (Exploratory Data Analysis)
```python
from kaggle_utils.visualization import *
from kaggle_utils.preprocessing import quick_info

# 1. Basic info
quick_info(train)

# 2. Missing values
plot_missing_values(train)

# 3. Target distribution
plot_target_distribution(train['price'], task='regression')

# 4. Feature distributions (train vs test)
plot_distributions(train, test, columns=['price', 'area', 'rooms'])

# 5. Correlations
plot_correlation_heatmap(train, top_n=20)
plot_correlation_with_target(train, 'price', top_n=20)
```

### Workflow 2: Model Analysis
```python
from kaggle_utils.models import LGBWrapper
from kaggle_utils.visualization import *

# 1. Train model
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train, X_test)

# 2. Feature importance
plot_feature_importance(lgb.models[0], X_train.columns, top_n=20)

# 3. Learning curves
plot_learning_curves(lgb.models[0], X_train, y_train, cv=5)

# 4. Predictions analysis
plot_predictions(y_val, lgb.predict(X_val))
```

### Workflow 3: Model Comparison
```python
from kaggle_utils.models import LGBWrapper, XGBWrapper, RandomForestWrapper
from kaggle_utils.visualization import plot_feature_importance_comparison

# Train models
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train)

xgb = XGBWrapper(n_splits=5)
xgb.train(X_train, y_train)

rf = RandomForestWrapper(n_splits=5)
rf.train(X_train, y_train)

# Compare feature importance
plot_feature_importance_comparison(
    [lgb.models[0], xgb.models[0], rf.models[0]],
    X_train.columns,
    ['LightGBM', 'XGBoost', 'RandomForest']
)
```

### Workflow 4: Classification Full Analysis
```python
from kaggle_utils.visualization import *

# 1. Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# 2. Confusion matrix
plot_confusion_matrix(y_val, y_pred, normalize=True)

# 3. ROC curve
plot_roc_curve(y_val, y_pred_proba)

# 4. Precision-Recall curve
plot_precision_recall_curve(y_val, y_pred_proba)

# 5. Feature importance
plot_feature_importance(model, X_train.columns, top_n=20)
```

---

## 🎨 Customization Examples

### Save Plots
```python
# บันทึกรูป
plot_feature_importance(
    model, 
    features,
    save_path='feature_importance.png'
)

plot_correlation_heatmap(
    train,
    save_path='correlation.png'
)
```

### Custom Figure Size
```python
# ปรับขนาด
plot_distributions(
    train, test,
    columns=['price', 'area'],
    figsize=(15, 8)
)

plot_correlation_heatmap(
    train,
    figsize=(16, 14)
)
```

### Custom Styling
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set custom style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Then use visualization functions
plot_distributions(train, test, columns=['price'])
```

---

## 🚨 Tips & Best Practices

### 1. 📊 Always Start with EDA
```python
# ก่อนทำอะไร ดู EDA ก่อนเสมอ!

# Missing values
plot_missing_values(train)

# Distributions
plot_distributions(train, test, columns=numeric_cols)

# Correlations
plot_correlation_heatmap(train, top_n=20)
```

### 2. 🎯 Feature Importance is Key
```python
# หลัง train model แล้ว ดู feature importance ทันที
lgb.train(X_train, y_train)
plot_feature_importance(lgb.models[0], X_train.columns, top_n=20)

# จะช่วยให้รู้ว่า features ไหนสำคัญ
# ลบ features ที่ไม่สำคัญออกได้
```

### 3. 🔍 Check Learning Curves
```python
# ใช้ learning curves ตรวจสอบ overfitting
plot_learning_curves(model, X_train, y_train, cv=5)

# ถ้าเห็น gap มาก → overfitting
# ถ้า scores ทั้งคู่ต่ำ → underfitting
```

### 4. 📈 Compare Train vs Test
```python
# เปรียบเทียบ distributions เสมอ!
plot_distributions(train, test, columns=['price', 'area'])

# ถ้า distributions ต่างกันมาก → ระวัง!
# อาจต้อง handle ด้วย domain adaptation
```

### 5. 🎨 Save Important Plots
```python
# บันทึก plots สำคัญไว้
plot_feature_importance(
    model, features,
    save_path='reports/feature_importance.png'
)

plot_learning_curves(
    model, X, y,
    save_path='reports/learning_curves.png'
)
```

### 6. 🔄 Iterate and Improve
```python
# Workflow: Plot → Analyze → Improve → Plot again

# Round 1
plot_feature_importance(model, features)
# เห็นว่า feature X ไม่สำคัญ → ลบออก

# Round 2
X_selected = X.drop(['feature_X'], axis=1)
model.fit(X_selected, y)
plot_feature_importance(model, X_selected.columns)
# ดีขึ้น! → Continue
```

---

## 📊 Plot Types Reference

| Plot Type | Function | Best For |
|-----------|----------|----------|
| Feature Importance | `plot_feature_importance()` | Feature selection |
| Distributions | `plot_distributions()` | Data drift detection |
| Correlation | `plot_correlation_heatmap()` | Feature engineering |
| Learning Curves | `plot_learning_curves()` | Overfitting detection |
| Confusion Matrix | `plot_confusion_matrix()` | Classification eval |
| ROC Curve | `plot_roc_curve()` | Threshold selection |
| Predictions | `plot_predictions()` | Regression eval |
| Missing Values | `plot_missing_values()` | Data quality check |

---

## 🎯 When to Use What?

### During EDA:
```python
plot_missing_values(train)
plot_target_distribution(y_train, task='regression')
plot_distributions(train, test, columns=numeric_cols)
plot_correlation_heatmap(train)
```

### During Feature Engineering:
```python
plot_correlation_with_target(train, 'price', top_n=20)
plot_distributions(train, test, columns=new_features)
```

### During Model Training:
```python
plot_learning_curves(model, X_train, y_train, cv=5)
plot_feature_importance(model, features, top_n=20)
```

### During Model Evaluation:
```python
# Regression
plot_predictions(y_val, y_pred)

# Classification
plot_confusion_matrix(y_val, y_pred)
plot_roc_curve(y_val, y_pred_proba)
plot_precision_recall_curve(y_val, y_pred_proba)
```

### For Final Report:
```python
# บันทึกทุก plots
plot_feature_importance(model, features, save_path='report/importance.png')
plot_learning_curves(model, X, y, save_path='report/learning.png')
plot_predictions(y_val, y_pred, save_path='report/predictions.png')
```

---

## 🔗 Integration Example

```python
from kaggle_utils import *
from kaggle_utils.visualization import *
from kaggle_utils.models import LGBWrapper
from kaggle_utils.diagnostics import quick_diagnosis

# 1. Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. EDA
plot_missing_values(train)
plot_target_distribution(train['price'], task='regression')
plot_distributions(train, test, columns=['price', 'area', 'rooms'])
plot_correlation_heatmap(train, top_n=20)

# 3. Diagnose
report = quick_diagnosis(train, 'price')

# 4. Train
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train, X_test)

# 5. Analyze
plot_feature_importance(lgb.models[0], X_train.columns, top_n=20)
plot_learning_curves(lgb.models[0], X_train, y_train, cv=5)
plot_predictions(y_val, lgb.predict(X_val))

# 6. Submit
create_submission(test['id'], lgb.test_predictions, 'submission.csv')
```

---

**💡 Pro Tip:** ใช้ `save_path` parameter เพื่อบันทึก plots ทั้งหมดไว้ในโฟลเดอร์ reports/ สำหรับทำรายงาน!
