# 🏥 Diagnostics Module - คู่มือการใช้งาน

เหมาะสำหรับมือใหม่ ML ที่ต้องการตรวจสอบคุณภาพข้อมูลและวินิจฉัยปัญหา

---

## 📚 ฟังก์ชันหลักที่มี

### 1. 🔍 **Data Quality Check**
```python
from kaggle_utils.diagnostics import check_data_quality

# ตรวจสอบคุณภาพข้อมูลโดยรวม
report = check_data_quality(train_df, target_col='price', show_details=True)
```

**ตรวจสอบ:**
- ✅ Missing values (features ที่มี missing มากกว่า 50%)
- ✅ Constant features (ไม่มีความแปรปรวน)
- ✅ Duplicate features (correlation > 0.95)
- ✅ High cardinality categorical features
- ✅ Target distribution และ class imbalance

**Output:**
```
🔍 DATA QUALITY CHECK
=================================================================
⚠️  3 features มี missing values > 50%
⚠️  2 features มีค่าคงที่ (ไม่มีความแปรปรวน)

💡 Suggestions:
   1. พิจารณาลบ features: ['col_A', 'col_B', 'col_C']
   2. ควรลบ constant features: ['col_X', 'col_Y']
```

---

### 2. 🚨 **Data Leakage Detection**
```python
from kaggle_utils.diagnostics import detect_leakage

# ตรวจจับ data leakage
suspicious = detect_leakage(train_df, target_col='price', test_df=test_df, threshold=0.95)
```

**ตรวจสอบ:**
- ✅ Features ที่มีความสัมพันธ์สูงผิดปกติกับ target (>0.95)
- ✅ Mutual Information ที่สูงผิดปกติ
- ✅ Train-Test distribution differences
- ✅ Temporal leakage (จาก date columns)

**Output:**
```
⚠️  พบ 2 features ที่มีความสัมพันธ์สูงผิดปกติกับ target (>0.95):
   - feature_A: 0.9823
   - feature_B: 0.9756

❌ อาจเป็น Data Leakage! ตรวจสอบ features เหล่านี้
```

---

### 3. 🔗 **Multicollinearity Check**
```python
from kaggle_utils.diagnostics import check_multicollinearity

# ตรวจสอบ features ที่มีความสัมพันธ์กันสูง
high_corr = check_multicollinearity(X_train, threshold=0.8)
```

**Output:**
```
⚠️  พบ 15 feature pairs ที่มีความสัมพันธ์สูง (>0.8)

💡 คำแนะนำ:
   1. พิจารณาลบ features ที่ซ้ำซ้อน
   2. ใช้ PCA หรือ feature selection
   3. ใช้ regularization (Ridge/Lasso)
```

---

### 4. 🤖 **Model Recommendation**
```python
from kaggle_utils.diagnostics import suggest_models

# แนะนำโมเดลที่เหมาะสม
recommendations = suggest_models(train_df, target_col='price', task='auto')
```

**แนะนำตาม:**
- 📊 Dataset size (small/medium/large)
- 📈 Sample/Feature ratio
- 📝 Number of categorical features
- 🎯 Task type (regression/classification)

**Output:**
```
📊 Dataset Info:
   - Task: REGRESSION
   - Samples: 5,000
   - Features: 50
   - Sample/Feature Ratio: 100.0

🎯 Recommended Models:
   1. 🥇 LightGBM / XGBoost
   2. 🥈 Random Forest
   3. 🥉 CatBoost (ถ้ามี categorical features เยอะ)
```

---

### 5. 📉 **Overfitting Detection**
```python
from kaggle_utils.diagnostics import detect_overfitting

# ตรวจจับ overfitting/underfitting
result = detect_overfitting(
    model=rf_model, 
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    task='regression'
)
```

**Output:**
```
📊 Performance Metrics:
   Train RMSE: 1234.56
   Val RMSE: 2345.67
   Gap: 1111.11 (90.0%)

❌ OVERFITTING detected!
💡 คำแนะนำ:
   1. เพิ่มข้อมูล training
   2. ใช้ regularization (Ridge, Lasso)
   3. ลด model complexity (max_depth, n_estimators)
   4. ใช้ early stopping
```

---

### 6. 📈 **Learning Curve Analysis**
```python
from kaggle_utils.diagnostics import plot_learning_curve

# Plot learning curve
plot_learning_curve(model, X_train, y_train, cv=5)
```

**แสดง:**
- 📊 Training score vs Validation score
- 📈 Score changes กับขนาด training set
- 🔍 วินิจฉัย overfitting/underfitting อัตโนมัติ

---

### 7. 🏥 **Quick Diagnosis (All-in-One)**
```python
from kaggle_utils.diagnostics import quick_diagnosis

# วินิจฉัยด่วน - รวมทุกอย่างในฟังก์ชันเดียว!
report = quick_diagnosis(
    train_df=train_df,
    target_col='price',
    test_df=test_df,
    model=rf_model,  # optional
    X_val=X_val,     # optional
    y_val=y_val      # optional
)
```

**ตรวจสอบทั้งหมด:**
1. ✅ Data Quality
2. ✅ Data Leakage
3. ✅ Multicollinearity
4. ✅ Model Recommendation
5. ✅ Overfitting Check (ถ้ามี model)

**Output:**
```
🏥 QUICK DIAGNOSIS - Comprehensive Data & Model Check
=================================================================

[... รายงานทั้งหมด ...]

📋 FINAL REPORT
=================================================================
⚠️  พบปัญหาทั้งหมด: 5

🔴 Priority Issues:
   1. ลบ constant features
   2. ตรวจสอบ data leakage features
   3. จัดการ multicollinearity

🎯 Recommended Next Steps:
   1. 🥇 LightGBM / XGBoost
   2. ใช้ cross-validation เพื่อประเมินโมเดล
   3. Plot learning curves เพื่อตรวจสอบ overfitting
```

---

### 8. 🎯 **Feature Importance Analysis**
```python
from kaggle_utils.diagnostics import analyze_feature_importance_detailed

# วิเคราะห์ feature importance แบบละเอียด
importance_df = analyze_feature_importance_detailed(
    model=trained_model,
    X=X_train,
    y=y_train,
    feature_names=X_train.columns,
    top_n=20
)
```

**แสดง:**
- 📊 Built-in feature importance
- 🔄 Permutation importance (แม่นยำกว่า)
- 💡 แนะนำ features ที่ควรลบ

---

## 💡 ตัวอย่างการใช้งานจริง

### Example 1: เริ่มต้นโปรเจคใหม่
```python
from kaggle_utils.diagnostics import quick_diagnosis
from kaggle_utils.preprocessing import reduce_mem_usage

# 1. โหลดข้อมูล
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. ลด memory
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# 3. วินิจฉัยด่วน
report = quick_diagnosis(
    train_df=train,
    target_col='target',
    test_df=test
)

# 4. ดำเนินการตามคำแนะนำ
# - ลบ constant features
# - ตรวจสอบ data leakage
# - เลือกโมเดลตามที่แนะนำ
```

### Example 2: ตรวจสอบก่อน Submit
```python
from kaggle_utils.diagnostics import (
    detect_leakage,
    detect_overfitting,
    plot_learning_curve
)
from sklearn.model_selection import train_test_split

# 1. Split data
X = train.drop(columns=['target'])
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. ตรวจสอบ data leakage
suspicious = detect_leakage(train, 'target', test)

# 3. Train model
model.fit(X_train, y_train)

# 4. ตรวจสอบ overfitting
overfitting_report = detect_overfitting(model, X_train, y_train, X_val, y_val)

# 5. Plot learning curve
plot_learning_curve(model, X, y, cv=5)

# ถ้าไม่มีปัญหา → Submit!
# ถ้ามีปัญหา → แก้ไขตามคำแนะนำ
```

### Example 3: Feature Engineering Workflow
```python
from kaggle_utils.diagnostics import (
    check_multicollinearity,
    analyze_feature_importance_detailed
)

# 1. สร้าง features ใหม่
train['feature_new'] = train['a'] * train['b']

# 2. ตรวจสอบ multicollinearity
high_corr = check_multicollinearity(train.drop(columns=['target']))

# 3. Train model
model.fit(X_train, y_train)

# 4. วิเคราะห์ feature importance
importance_df = analyze_feature_importance_detailed(
    model, X_train, y_train, X_train.columns
)

# 5. ลบ features ที่ไม่สำคัญ
features_to_keep = importance_df[importance_df['importance'] > 0.001]['feature'].tolist()
X_train_selected = X_train[features_to_keep]
```

---

## 🎯 เมื่อไหร่ควรใช้ฟังก์ชันไหน?

| สถานการณ์ | ฟังก์ชันที่แนะนำ |
|-----------|------------------|
| 🆕 เริ่มต้นโปรเจคใหม่ | `quick_diagnosis()` |
| 🔍 ตรวจสอบคุณภาพข้อมูล | `check_data_quality()` |
| 🚨 สงสัยว่ามี data leakage | `detect_leakage()` |
| 🤖 ไม่รู้จะใช้โมเดลไหน | `suggest_models()` |
| 📊 Train/Val score ห่างกันมาก | `detect_overfitting()` |
| 📈 ต้องการดู learning curve | `plot_learning_curve()` |
| 🎯 เลือก features | `analyze_feature_importance_detailed()` |
| 🔗 Features หลายตัวคล้ายกัน | `check_multicollinearity()` |

---

## ⚠️ คำเตือนสำคัญ

### Data Leakage คืออะไร?
- Features ที่มีข้อมูลจากอนาคต (รู้คำตอบก่อน)
- Features ที่คำนวณจาก target โดยตรง
- Test data รั่วไหลเข้า training set

**ตัวอย่าง:**
```python
# ❌ WRONG - Data Leakage!
train['price_mean'] = train.groupby('category')['price'].transform('mean')
# ← ใช้ค่าเฉลี่ยทั้งหมด รวมถึง future data

# ✅ CORRECT
train['price_mean'] = train.groupby('category')['price'].transform(
    lambda x: x.expanding().mean().shift()
)
# ← ใช้แค่ข้อมูลก่อนหน้า
```

### Overfitting vs Underfitting

**Overfitting:**
- Train score ดีมาก แต่ Val score แย่
- Model จำข้อมูล training แทนที่จะเรียนรู้ pattern
- แก้: Regularization, เพิ่มข้อมูล, ลด complexity

**Underfitting:**
- ทั้ง Train และ Val score แย่ทั้งคู่
- Model ง่ายเกินไป จับ pattern ไม่ได้
- แก้: เพิ่ม complexity, เพิ่ม features

---

## 🚀 Integration กับ Workflow

```python
from kaggle_utils import *
from kaggle_utils.diagnostics import quick_diagnosis

# 1. Setup
setup_colab()

# 2. Load & Inspect
train = pd.read_csv('train.csv')
quick_info(train)

# 3. Comprehensive Diagnosis
report = quick_diagnosis(train, target_col='target', test_df=test)

# 4. Clean Data (ตามคำแนะนำ)
train = train.drop(columns=report['quality_report']['constant_features'])

# 5. Train Models
rf = RandomForestWrapper(n_splits=5)
rf.train(X_train, y_train, X_test)

# 6. Check Overfitting
detect_overfitting(rf.models[0], X_train, y_train, X_val, y_val)

# 7. Submit
create_submission(test['id'], rf.test_predictions, 'submission.csv')
```

---

## 📖 เพิ่มเติม

ดู examples ใน `examples/` folder:
- `example_diagnostics.ipynb` - ตัวอย่างการใช้งานแบบละเอียด
- `example_beginner_workflow.ipynb` - Workflow สำหรับมือใหม่

---

**💡 Pro Tip:** ใช้ `quick_diagnosis()` ก่อนเสมอเมื่อเริ่มโปรเจคใหม่!
