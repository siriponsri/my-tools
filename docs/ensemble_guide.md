# 📖 Ensemble Module - คู่มือการใช้งาน

## 🎯 Ensemble Methods ที่มี

### 1. **WeightedEnsemble** - Weighted Average
- รวม predictions ด้วยน้ำหนักที่กำหนด
- ง่าย รวดเร็ว มีประสิทธิภาพ

### 2. **StackingEnsemble** - Stacking with Meta-Learner
- ใช้ predictions จาก base models เป็น features
- Train meta-model เพื่อทำนายขั้นสุดท้าย
- แม่นยำที่สุด แต่ใช้เวลานาน

### 3. **create_voting_ensemble()** - Voting
- Majority vote (hard) หรือ average probabilities (soft)
- sklearn's VotingClassifier/VotingRegressor

### 4. **blend_predictions()** - Flexible Blending
- Average, Rank average, Geometric mean
- ใช้ได้กับ predictions ที่มีอยู่แล้ว

### 5. **optimize_blend_weights()** - Optimal Weights
- หาน้ำหนักที่ดีที่สุดอัตโนมัติ
- Grid search หรือ Random search

### 6. **DynamicEnsemble** - Dynamic Model Selection
- เลือกโมเดลที่ดีที่สุดสำหรับแต่ละตัวอย่าง
- Advanced technique

---

## 💡 ตัวอย่างการใช้งาน

### Example 1: Weighted Ensemble (แนะนำสำหรับเริ่มต้น)

```python
from kaggle_utils.ensemble import WeightedEnsemble
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# สร้าง models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
ridge = Ridge(alpha=1.0)

# Weighted Ensemble
ensemble = WeightedEnsemble(
    models=[rf, gbm, ridge],
    weights=[0.5, 0.3, 0.2],  # ให้น้ำหนัก RF มากที่สุด
    task='regression',
    verbose=True
)

# Train
ensemble.fit(X_train, y_train)

# Predict
predictions = ensemble.predict(X_test)
```

**Output:**
```
============================================================
🎯 Training Weighted Ensemble (3 models)
============================================================
Weights: ['0.500', '0.300', '0.200']
Training Models: 100%|██████████████████████████| 3/3 [00:15<00:00]

✅ All models trained successfully!
```

---

### Example 2: Stacking Ensemble (แม่นยำที่สุด!)

```python
from kaggle_utils.ensemble import StackingEnsemble
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso

# Base models (level 0)
base_models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    ExtraTreesRegressor(n_estimators=100, random_state=42),
    Lasso(alpha=0.1)
]

# Meta model (level 1)
meta_model = Ridge(alpha=1.0)

# Stacking
stacking = StackingEnsemble(
    base_models=base_models,
    meta_model=meta_model,
    cv=5,
    task='regression',
    verbose=True
)

# Train (ใช้เวลานานกว่า แต่แม่นกว่า!)
stacking.fit(X_train, y_train)

# Predict
predictions = stacking.predict(X_test)
```

**Output:**
```
============================================================
🎯 Training Stacking Ensemble
============================================================
Base Models: 3
Meta Model: Ridge
CV Folds: 5

📊 Step 1/3: Creating meta-features with 5-fold CV...
Training Base Models: 100%|████████████████| 3/3 [02:30<00:00]
  Model 1 Folds: 100%|████████████| 5/5
  Model 2 Folds: 100%|████████████| 5/5
  Model 3 Folds: 100%|████████████| 5/5

📊 Step 2/3: Training meta model...
✅ Meta model trained!

📊 Step 3/3: Retraining base models on full data...
Retraining Base Models: 100%|████████████| 3/3 [00:45<00:00]

✅ Stacking ensemble training complete!
```

---

### Example 3: Voting Ensemble

```python
from kaggle_utils.ensemble import create_voting_ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Models
models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    SVC(probability=True, random_state=42)
]

# Voting Ensemble
voting = create_voting_ensemble(
    models=models,
    voting='soft',  # 'soft' = average probabilities, 'hard' = majority vote
    weights=[0.5, 0.3, 0.2],
    task='classification',
    verbose=True
)

# Train
voting.fit(X_train, y_train)

# Predict
predictions = voting.predict(X_test)
```

**Output:**
```
✅ Created Voting Classifier (soft voting) with 3 models
   Weights: [0.5, 0.3, 0.2]
```

---

### Example 4: Blend Predictions (ใช้กับ predictions ที่มีอยู่แล้ว)

```python
from kaggle_utils.ensemble import blend_predictions

# สมมติว่ามี predictions จาก 3 models แล้ว
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# Method 1: Average (แนะนำ)
final_avg = blend_predictions(
    [pred1, pred2, pred3],
    weights=[0.5, 0.3, 0.2],
    method='average',
    verbose=True
)

# Method 2: Rank Average (ดีถ้า predictions มี scale ต่างกัน)
final_rank = blend_predictions(
    [pred1, pred2, pred3],
    weights=[0.5, 0.3, 0.2],
    method='rank',
    verbose=True
)

# Method 3: Geometric Mean (ดีสำหรับ positive predictions)
final_geo = blend_predictions(
    [pred1, pred2, pred3],
    weights=[0.5, 0.3, 0.2],
    method='geometric',
    verbose=True
)
```

**Output:**
```
============================================================
🔀 Blending 3 predictions
============================================================
Method: average
Weights: ['0.500', '0.300', '0.200']
✅ Blending complete!
   Shape: (10000,)
```

---

### Example 5: Optimize Blend Weights (หาน้ำหนักที่ดีที่สุด!)

```python
from kaggle_utils.ensemble import optimize_blend_weights, blend_predictions

# Step 1: Get predictions on validation set
pred1_val = model1.predict(X_val)
pred2_val = model2.predict(X_val)
pred3_val = model3.predict(X_val)

# Step 2: หาน้ำหนักที่ดีที่สุด
optimal_weights = optimize_blend_weights(
    predictions_list=[pred1_val, pred2_val, pred3_val],
    y_true=y_val,
    metric='rmse',
    method='grid',  # หรือ 'random'
    n_trials=100,
    verbose=True
)

# Step 3: ใช้น้ำหนักที่ดีที่สุดกับ test set
pred1_test = model1.predict(X_test)
pred2_test = model2.predict(X_test)
pred3_test = model3.predict(X_test)

final = blend_predictions(
    [pred1_test, pred2_test, pred3_test],
    weights=optimal_weights,
    method='average'
)
```

**Output:**
```
============================================================
🔍 Optimizing blend weights (grid search)
============================================================
Models: 3
Metric: RMSE
Testing 1331 weight combinations...
Grid Search: 100%|██████████████| 1331/1331 [00:15<00:00] Best Score: 1234.5678

✅ Optimization complete!
   Best RMSE: 1234.5678
   Optimal Weights: ['0.600', '0.300', '0.100']
```

---

### Example 6: Dynamic Ensemble (Advanced)

```python
from kaggle_utils.ensemble import DynamicEnsemble
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

# Base models
models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    Ridge(alpha=1.0)
]

# Selection model (เลือกว่าจะใช้ model ไหน)
selection_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Dynamic Ensemble
dynamic = DynamicEnsemble(
    models=models,
    selection_model=selection_model,
    task='regression',
    verbose=True
)

# Train
dynamic.fit(X_train, y_train)

# Predict (แต่ละตัวอย่างอาจใช้คนละ model!)
predictions = dynamic.predict(X_test)
```

---

## 🎯 Complete Workflow Example

```python
from kaggle_utils.models import LGBWrapper, XGBWrapper, RandomForestWrapper
from kaggle_utils.ensemble import blend_predictions, optimize_blend_weights
from kaggle_utils.utils import create_submission
from sklearn.model_selection import train_test_split

# === 1. SPLIT DATA ===
X = train.drop(columns=['target'])
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. TRAIN MULTIPLE MODELS ===
print("🚀 Training Models...")

# LightGBM
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train, X_test=test)

# XGBoost
xgb = XGBWrapper(n_splits=5)
xgb.train(X_train, y_train, X_test=test)

# Random Forest
rf = RandomForestWrapper(n_splits=5)
rf.train(X_train, y_train, X_test=test)

# === 3. OPTIMIZE BLEND WEIGHTS ON VALIDATION SET ===
print("\n🔍 Optimizing blend weights...")

# Get validation predictions
lgb_val = lgb.predict(X_val)
xgb_val = xgb.predict(X_val)
rf_val = rf.predict(X_val)

# Find optimal weights
optimal_weights = optimize_blend_weights(
    [lgb_val, xgb_val, rf_val],
    y_val,
    metric='rmse',
    method='grid'
)

# === 4. BLEND TEST PREDICTIONS ===
print("\n🔀 Blending final predictions...")
final_pred = blend_predictions(
    [lgb.test_predictions, xgb.test_predictions, rf.test_predictions],
    weights=optimal_weights,
    method='average'
)

# === 5. CREATE SUBMISSION ===
create_submission(test['id'], final_pred, 'submission.csv')
```

---

## 📊 เปรียบเทียบ Ensemble Methods

| Method | Complexity | Training Time | Accuracy | When to Use |
|--------|-----------|---------------|----------|-------------|
| **Weighted Average** | ⭐ | ⚡⚡⚡ | ⭐⭐⭐ | Quick baseline, simple |
| **Voting** | ⭐⭐ | ⚡⚡ | ⭐⭐⭐ | Classification, sklearn integration |
| **Blending** | ⭐ | ⚡⚡⚡ | ⭐⭐⭐ | When you have predictions already |
| **Stacking** | ⭐⭐⭐ | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy, have time |
| **Dynamic** | ⭐⭐⭐⭐ | ⚡ | ⭐⭐⭐⭐ | Advanced, different models excel differently |

---

## 🚨 Tips & Best Practices

### 1. 📊 **Diverse Models Work Best**
```python
# ❌ BAD - Similar models
models = [
    RandomForestRegressor(n_estimators=100),
    RandomForestRegressor(n_estimators=200),
    RandomForestRegressor(n_estimators=300)
]

# ✅ GOOD - Diverse models
models = [
    RandomForestRegressor(),      # Tree-based
    Ridge(),                       # Linear
    GradientBoostingRegressor()   # Boosting
]
```

### 2. ⚖️ **Start with Equal Weights**
```python
# เริ่มต้นด้วยน้ำหนักเท่ากัน
ensemble = WeightedEnsemble(models, weights=None)  # Auto: [0.33, 0.33, 0.34]

# แล้วค่อย optimize
optimal_weights = optimize_blend_weights(predictions, y_val)
```

### 3. 🎯 **Use Validation Set for Weight Optimization**
```python
# ❌ WRONG - Optimize on training set (overfitting!)
optimal_weights = optimize_blend_weights(train_predictions, y_train)

# ✅ CORRECT - Optimize on validation set
optimal_weights = optimize_blend_weights(val_predictions, y_val)

# Use for test
final = blend_predictions(test_predictions, weights=optimal_weights)
```

### 4. 🔄 **Stacking Tips**
```python
# สำหรับ Stacking:
# - Base models ควรหลากหลาย
# - Meta model ควรเรียบง่าย (Ridge, Lasso)
# - ใช้ CV ≥ 5 folds

stacking = StackingEnsemble(
    base_models=[rf, gbm, lasso],  # หลากหลาย
    meta_model=Ridge(alpha=1.0),   # เรียบง่าย
    cv=5                           # อย่างน้อย 5
)
```

### 5. 📈 **Diminishing Returns**
```python
# 2-3 models มักดีพอแล้ว
# เพิ่มเกิน 5 models มักไม่ได้ประโยชน์มากขึ้น

# Good
ensemble = WeightedEnsemble([lgb, xgb, rf])

# Overkill (อาจไม่คุ้ม)
ensemble = WeightedEnsemble([lgb, xgb, rf, cat, gbm, et, ada, svm])
```

---

## 🎨 Advanced Techniques

### Multi-Level Stacking
```python
# Level 0: Base models
rf = RandomForestRegressor()
gbm = GradientBoostingRegressor()
lasso = Lasso()

# Level 1: Meta model (also an ensemble!)
meta_models = [Ridge(), ElasticNet()]
meta_ensemble = WeightedEnsemble(meta_models, weights=[0.6, 0.4])

# Stacking
stacking = StackingEnsemble(
    base_models=[rf, gbm, lasso],
    meta_model=meta_ensemble,  # Ensemble as meta!
    cv=5
)
```

### Cascading Ensemble
```python
# First layer
lgb = LGBWrapper(n_splits=5)
lgb.train(X_train, y_train)

# Second layer (use OOF predictions as features)
X_train_meta = np.column_stack([X_train, lgb.oof_predictions.reshape(-1, 1)])

xgb = XGBWrapper(n_splits=5)
xgb.train(X_train_meta, y_train)
```

---

## 📌 Quick Reference

| Task | Function | Code |
|------|----------|------|
| Simple averaging | `WeightedEnsemble` | `WeightedEnsemble(models, weights=[0.5, 0.5])` |
| Best accuracy | `StackingEnsemble` | `StackingEnsemble(base_models, meta_model, cv=5)` |
| Classification vote | `create_voting_ensemble` | `create_voting_ensemble(models, voting='soft')` |
| Blend predictions | `blend_predictions` | `blend_predictions([pred1, pred2], weights=[0.6, 0.4])` |
| Find best weights | `optimize_blend_weights` | `optimize_blend_weights([pred1, pred2], y_val)` |

---

**💡 Pro Tip:** เริ่มจาก simple ensemble (Weighted/Blending) ก่อน ถ้าต้องการความแม่นยำสูงสุดค่อยใช้ Stacking!
