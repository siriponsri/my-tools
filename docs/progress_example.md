# 📊 Progress Bar Examples

## 🎯 สิ่งที่เพิ่มเข้ามา

### 1. **tqdm Progress Bars** ในทุก Wrapper
- แสดง progress สำหรับแต่ละ fold
- แสดง current score ใน postfix
- สวยงาม ดูง่าย รู้ว่าเหลืออีกเท่าไร

### 2. **Verbose Control**
- `verbose=True` (default) - แสดง progress bar
- `verbose=False` - ไม่แสดง (สำหรับ production)

---

## 💡 ตัวอย่าง Output

### Example 1: LightGBM Training
```python
from kaggle_utils.models import LGBWrapper

lgb = LGBWrapper(n_splits=5, verbose=True)
lgb.train(X_train, y_train, X_test)
```

**Output:**
```
============================================================
🚀 Training LightGBM with 5-Fold CV
============================================================
🌲 LightGBM Training:  60%|███████████████        | 3/5 [02:30<01:40] Fold: 3, RMSE: 1234.5678
```

### Example 2: Random Forest Training
```python
from kaggle_utils.models import RandomForestWrapper

rf = RandomForestWrapper(n_splits=5, verbose=True)
rf.train(X_train, y_train, X_test)
```

**Output:**
```
============================================================
🚀 Training RandomForestRegressor with 5-Fold CV
============================================================
Training Folds:  80%|████████████████████████      | 4/5 Fold: 4, RMSE: 1235.1234
```

### Example 3: Quick Model Comparison
```python
from kaggle_utils.models import quick_model_comparison

results = quick_model_comparison(X_train, y_train, cv=5, verbose=True)
```

**Output:**
```
🔄 Comparing regression models...
============================================================
Testing Models:  56%|████████████████▌             | 5/9 Current: Random Forest

============================================================
✅ Best model: Random Forest
   RMSE: 1235.4567
```

### Example 4: Scaler Comparison
```python
from kaggle_utils.models import compare_scalers
from sklearn.linear_model import Ridge

best_scaler, results = compare_scalers(X, y, Ridge(), cv=5, verbose=True)
```

**Output:**
```
🔄 Comparing scalers...
============================================================
Testing Scalers: 100%|██████████████████████████████| 5/5 Current: MaxAbsScaler

============================================================
✅ Best scaler: StandardScaler
   RMSE: 1445.3421
   StandardScaler      : 1445.3421
   RobustScaler        : 1446.1234
   MinMaxScaler        : 1448.5678
   MaxAbsScaler        : 1449.2341
   No Scaling          : 1450.9876
```

---

## 🎨 Progress Bar Styles

### Style 1: Fold Training (สำหรับ sklearn)
```
Training Folds:  60%|██████████████▌        | 3/5 Fold: 3, RMSE: 1234.5678
```

### Style 2: Gradient Boosting (LGB/XGB/Cat)
```
🌲 LightGBM Training:  80%|████████████████████▌ | 4/5 [03:20<00:50] Fold: 4, RMSE: 1235.1234
```

### Style 3: Model Comparison
```
Testing Models:  67%|████████████████▊       | 6/9 Current: Gradient Boosting
```

---

## 🔧 การใช้งาน

### เปิด Progress Bar (Default)
```python
# แบบนี้จะมี progress bar
lgb = LGBWrapper(n_splits=5, verbose=True)  # หรือไม่ใส่ก็ได้ (default=True)
lgb.train(X, y, X_test)
```

### ปิด Progress Bar
```python
# แบบนี้ไม่มี progress bar (เงียบๆ)
lgb = LGBWrapper(n_splits=5, verbose=False)
lgb.train(X, y, X_test)
```

### Progress Bar ใน Loop
```python
from kaggle_utils.models import LGBWrapper, XGBWrapper, RandomForestWrapper

models = {
    'LightGBM': LGBWrapper(n_splits=5),
    'XGBoost': XGBWrapper(n_splits=5),
    'RandomForest': RandomForestWrapper(n_splits=5)
}

# แต่ละ model จะมี progress bar ของตัวเอง
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")
    model.train(X_train, y_train, X_test)
```

**Output:**
```
============================================================
Training LightGBM
============================================================
============================================================
🚀 Training LightGBM with 5-Fold CV
============================================================
🌲 LightGBM Training: 100%|██████████████████████| 5/5 [04:15<00:00] Fold: 5, RMSE: 1234.9876

📊 Overall RMSE: 1235.3802 (±0.6234)

============================================================
Training XGBoost
============================================================
============================================================
🚀 Training XGBoost with 5-Fold CV
============================================================
🎯 XGBoost Training: 100%|███████████████████████| 5/5 [05:30<00:00] Fold: 5, RMSE: 1233.5432

📊 Overall RMSE: 1234.1234 (±0.5678)
```

---

## 🌟 Features

### 1. **Real-time Progress**
- รู้ว่าอยู่ fold ไหน (3/5)
- รู้ว่าใช้เวลาไปเท่าไร (03:20)
- รู้ว่าเหลืออีกนานแค่ไหน (<00:50)

### 2. **Current Metrics**
- แสดง score ของ fold ปัจจุบัน
- RMSE/AUC อัปเดตทันที
- เห็นได้ชัดเจนว่า model กำลังทำงาน

### 3. **Clean Output**
- ซ่อน verbose ของ LightGBM/XGBoost/CatBoost
- แสดงแค่ progress bar เดียว
- ไม่รก ดูง่าย

### 4. **Jupyter/Colab Friendly**
- ใช้ `tqdm.auto` จะเลือก style ให้อัตโนมัติ
- ใน Jupyter/Colab จะแสดงแบบสวยงาม
- ใน terminal จะแสดงแบบ text-based

---

## 📊 เปรียบเทียบ: ก่อน vs หลัง

### ❌ ก่อน (ไม่มี Progress Bar)
```
Training Fold 1...
Training Fold 2...
Training Fold 3...
[รอนาน ไม่รู้ว่าถึงไหนแล้ว... 😴]
Training Fold 4...
Training Fold 5...
Done!
```

### ✅ หลัง (มี Progress Bar)
```
🌲 LightGBM Training:  60%|███████████████        | 3/5 [02:30<01:20] Fold: 3, RMSE: 1234.5678
                                                   ↑           ↑         ↑          ↑
                                          อยู่ fold 3/5   ใช้เวลา 2:30  เหลืออีก 1:20  Score ปัจจุบัน
```

---

## 🎯 Best Practices

### 1. **Development (verbose=True)**
```python
# ขณะ develop/experiment ใช้ verbose=True
lgb = LGBWrapper(verbose=True)  # เห็น progress
lgb.train(X, y, X_test)
```

### 2. **Production (verbose=False)**
```python
# ขณะ production/automation ใช้ verbose=False
lgb = LGBWrapper(verbose=False)  # เงียบๆ
lgb.train(X, y, X_test)
```

### 3. **Jupyter Notebook**
```python
# ใน notebook progress bar จะแสดงแบบสวยงาม
from kaggle_utils.models import LGBWrapper

lgb = LGBWrapper(n_splits=10)  # แม้ 10 folds ก็ไม่กลัว!
lgb.train(X_train, y_train, X_test)
```

---

## 💡 Tips

### Tip 1: แสดงว่า Cell ยังทำงานอยู่
```python
# Progress bar จะหมุนเรื่อยๆ แสดงว่ายังไม่หยุด
# ไม่ต้องกังวลว่า kernel crash หรือเปล่า
```

### Tip 2: ประมาณเวลาที่เหลือ
```python
# เห็นว่าเหลืออีก 5 นาที
# สามารถไปทำอย่างอื่นได้
```

### Tip 3: Debug Performance
```python
# ถ้า fold แรกใช้เวลานาน
# แสดงว่าทั้งหมดจะนานด้วย
# สามารถ interrupt และปรับ parameters
```

---

## 🔗 Dependencies

Progress bars ใช้ **tqdm** library:
```bash
pip install tqdm
```

หรือ
```bash
pip install kaggle-utils[full]  # มี tqdm รวมอยู่แล้ว
```

---

## 📌 Summary

| Feature | Status |
|---------|--------|
| Progress Bar | ✅ ใช้ได้ |
| Time Estimation | ✅ แสดงเวลาคงเหลือ |
| Current Metrics | ✅ แสดง score real-time |
| Jupyter/Colab | ✅ รองรับ |
| Clean Output | ✅ ไม่รก |
| Verbose Control | ✅ เปิด/ปิดได้ |

**💡 ตอนนี้ไม่ต้องกลัวว่า cell หยุดทำงานแล้ว เพราะมี progress bar บอกตลอดเวลา! 🎉**
