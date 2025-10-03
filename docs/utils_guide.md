# 🛠️ Utils Guide

คู่มือการใช้งาน Utility Functions สำหรับ Kaggle Competitions

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Loading & Saving](#data-loading--saving)
- [Kaggle Integration](#kaggle-integration)
- [Submission](#submission)
- [Timing & Profiling](#timing--profiling)
- [Memory Management](#memory-management)
- [Examples](#examples)

---

## Overview

Module นี้รวมฟังก์ชันช่วยเหลือต่างๆ สำหรับการแข่ง Kaggle เช่น:
- Setup environment (Colab, Kaggle)
- โหลดและบันทึกข้อมูล
- สร้างไฟล์ submission
- วัดเวลาและ memory usage

---

## Environment Setup

### 1. `setup_colab()`

ตั้งค่า Google Colab environment

```python
from kaggle_utils import setup_colab

# Setup Colab
setup_colab()

# ทำอะไร:
# - Mount Google Drive
# - ติดตั้ง kaggle-utils (ถ้ายังไม่มี)
```

**Output:**
```
============================================================
🚀 Setting up Google Colab
============================================================

📁 Mounting Google Drive...
✅ Drive mounted at: /content/drive

✅ kaggle-utils already installed

============================================================
✅ Colab setup complete!
============================================================
```

### 2. `setup_kaggle()`

ตั้งค่า Kaggle API

```python
from kaggle_utils import setup_kaggle

# Auto detect kaggle.json
setup_kaggle()

# หรือระบุ path เอง
setup_kaggle(kaggle_json_path='/path/to/kaggle.json')
```

**ทำอะไร:**
- สร้าง `~/.kaggle/` directory
- Copy `kaggle.json` ไปที่ถูกต้อง
- Set permissions (chmod 600)

**วิธีหา kaggle.json:**
1. ไปที่ https://www.kaggle.com/
2. คลิก Account → Create New API Token
3. ดาวน์โหลด `kaggle.json`

### 3. `check_environment()`

ตรวจสอบ environment ที่ใช้อยู่

```python
from kaggle_utils import check_environment

env = check_environment()
print(env)

# Output:
# {
#     'environment': 'colab',  # 'colab', 'kaggle', 'local'
#     'python_version': '3.10.12',
#     'platform': 'Linux',
#     'gpu_available': True,
#     'ram_total_gb': 12.7
# }
```

---

## Data Loading & Saving

### 1. `load_data()`

โหลดข้อมูลจากไฟล์ (รองรับหลาย format)

```python
from kaggle_utils import load_data

# CSV
df = load_data('train.csv')

# Parquet (เร็วกว่า CSV)
df = load_data('train.parquet')

# Excel
df = load_data('data.xlsx', sheet_name='Sheet1')

# JSON
df = load_data('data.json')

# Auto detect encoding
df = load_data('train.csv', encoding='auto')

# Show info
df = load_data('train.csv', show_info=True)
```

**Parameters:**
- `filepath` (str): Path ของไฟล์
- `encoding` (str): 'utf-8', 'latin1', 'auto'
- `show_info` (bool): แสดงข้อมูลหลัง load
- `**kwargs`: ส่งต่อไปยัง pandas.read_*

**Supported Formats:**
- `.csv` → `pd.read_csv()`
- `.parquet`, `.pq` → `pd.read_parquet()`
- `.xlsx`, `.xls` → `pd.read_excel()`
- `.json` → `pd.read_json()`
- `.feather` → `pd.read_feather()`
- `.pickle`, `.pkl` → `pd.read_pickle()`

### 2. `save_data()`

บันทึกข้อมูล (auto detect format จากชื่อไฟล์)

```python
from kaggle_utils import save_data

# CSV
save_data(df, 'output.csv')

# Parquet (แนะนำ - เร็วและเล็ก)
save_data(df, 'output.parquet')

# Excel
save_data(df, 'output.xlsx', sheet_name='Results')

# JSON
save_data(df, 'output.json')

# CSV โดยไม่มี index
save_data(df, 'output.csv', index=False)
```

**Tips:**
- ใช้ Parquet สำหรับไฟล์ใหญ่ (เร็วกว่า CSV 10-100x)
- ใช้ CSV สำหรับ submission

---

## Kaggle Integration

### 1. `download_kaggle_dataset()`

ดาวน์โหลด dataset จาก Kaggle

```python
from kaggle_utils import download_kaggle_dataset

# ดาวน์โหลด dataset
download_kaggle_dataset(
    dataset_name='house-prices-advanced-regression-techniques',
    path='./data'
)

# ดาวน์โหลด competition data
download_kaggle_dataset(
    competition='titanic',
    path='./data',
    competition=True
)
```

**Parameters:**
- `dataset_name` (str): ชื่อ dataset หรือ competition
- `path` (str): ที่จะเก็บไฟล์
- `competition` (bool): เป็น competition หรือ dataset
- `unzip` (bool): แตกไฟล์ zip อัตโนมัติ

**ตัวอย่าง:**
```python
# House Prices Competition
download_kaggle_dataset(
    competition='house-prices-advanced-regression-techniques',
    path='./data',
    competition=True,
    unzip=True
)

# หลังจากนี้จะมีไฟล์:
# ./data/train.csv
# ./data/test.csv
# ./data/sample_submission.csv
```

---

## Submission

### `create_submission()`

สร้างไฟล์ submission สำหรับ Kaggle

```python
from kaggle_utils import create_submission

# แบบง่าย
create_submission(
    ids=test_ids,
    predictions=predictions,
    filename='submission.csv'
)

# Custom column names
create_submission(
    ids=test['PassengerId'],
    predictions=predictions,
    filename='submission.csv',
    id_column='PassengerId',
    target_column='Survived'
)

# With info
create_submission(
    ids=test_ids,
    predictions=predictions,
    filename='submission.csv',
    show_sample=True  # แสดง 5 แถวแรก
)
```

**Output:**
```
✅ Submission file created: submission.csv
📊 Shape: (418, 2)
💾 Size: 6.2 KB

Sample:
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
3          895         0
4          896         1
```

**Default Column Names:**
- ID column: `'id'`
- Target column: `'target'`

---

## Timing & Profiling

### 1. `timer()` Decorator

Decorator สำหรับวัดเวลา

```python
from kaggle_utils import timer

@timer
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# ใช้งาน
model = train_model(X_train, y_train)

# Output:
# ⏱️  train_model took 5.23 seconds
```

### 2. `Timer` Class

Context manager สำหรับวัดเวลา

```python
from kaggle_utils import Timer

# แบบ context manager
with Timer("Loading data"):
    df = pd.read_csv('large_file.csv')

# Output:
# ⏱️  Loading data took 12.45 seconds

# แบบ manual
timer = Timer()
timer.start()

# ... do something ...

elapsed = timer.stop()
print(f"Elapsed: {elapsed:.2f} seconds")
```

**Advanced Usage:**
```python
from kaggle_utils import Timer

# วัดหลายขั้นตอน
timer = Timer("Full Pipeline")

timer.lap("Loading data")
df = load_data('train.csv')

timer.lap("Preprocessing")
df_processed = preprocess(df)

timer.lap("Training")
model.fit(X_train, y_train)

timer.lap("Predicting")
predictions = model.predict(X_test)

timer.stop()

# Output:
# ⏱️  Loading data: 2.3s
# ⏱️  Preprocessing: 5.1s
# ⏱️  Training: 45.2s
# ⏱️  Predicting: 1.2s
# ⏱️  Full Pipeline took 53.8 seconds
```

---

## Memory Management

### 1. `reduce_mem_usage()`

ลด memory usage ของ DataFrame

```python
from kaggle_utils import reduce_mem_usage

# ก่อน
print(f"Before: {df.memory_usage().sum() / 1024**2:.2f} MB")

# ลด memory
df = reduce_mem_usage(df, verbose=True)

# หลัง
print(f"After: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Output:
# Memory usage decreased from 450.32 MB to 112.58 MB (75.0% reduction)
```

**ทำอะไร:**
- แปลง int64 → int8/int16/int32 (ถ้าทำได้)
- แปลง float64 → float32 (ถ้าทำได้)
- ลด memory โดยไม่เสีย information

### 2. `memory_usage()`

ดู memory usage ของตัวแปร

```python
from kaggle_utils import memory_usage

# Single variable
print(memory_usage(df))
# Output: "df: 123.45 MB"

# Multiple variables
print(memory_usage(df, X_train, y_train, model))
# Output:
# df: 123.45 MB
# X_train: 45.67 MB
# y_train: 1.23 MB
# model: 15.89 MB
# Total: 186.24 MB
```

### 3. `estimate_file_size()`

ประมาณขนาดไฟล์ก่อนบันทึก

```python
from kaggle_utils import estimate_file_size

size_mb = estimate_file_size(df, format='csv')
print(f"Estimated CSV size: {size_mb:.2f} MB")

size_mb = estimate_file_size(df, format='parquet')
print(f"Estimated Parquet size: {size_mb:.2f} MB")
```

---

## Additional Utilities

### `set_seed()`

ตั้งค่า random seed สำหรับ reproducibility

```python
from kaggle_utils import set_seed

# Set seed ทุกอย่าง
set_seed(42)

# ตั้งค่า:
# - random
# - numpy
# - torch (ถ้ามี)
# - tensorflow (ถ้ามี)
```

### `notify()`

ส่ง notification เมื่อ training เสร็จ (สำหรับ Colab)

```python
from kaggle_utils import notify

# Train model (นานมาก)
model.fit(X_train, y_train)

# ส่ง notification
notify("Model training complete! 🎉")

# บน Colab จะได้ browser notification
```

---

## Examples

### Example 1: Complete Workflow

```python
from kaggle_utils import *

# 1. Setup (ถ้าใช้ Colab)
setup_colab()
setup_kaggle()

# 2. Download data
download_kaggle_dataset(
    competition='house-prices-advanced-regression-techniques',
    path='./data',
    competition=True
)

# 3. Load data
train = load_data('./data/train.csv', show_info=True)
test = load_data('./data/test.csv')

# 4. Reduce memory
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# 5. Train model
@timer
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X_train, y_train)

# 6. Predict
predictions = model.predict(X_test)

# 7. Create submission
create_submission(
    ids=test['Id'],
    predictions=predictions,
    filename='submission.csv',
    id_column='Id',
    target_column='SalePrice',
    show_sample=True
)

# 8. Notify (ถ้าใช้ Colab)
notify("Pipeline complete! 🎉")
```

### Example 2: Large Dataset Handling

```python
from kaggle_utils import *

# Check environment
env = check_environment()
print(f"RAM available: {env['ram_total_gb']:.1f} GB")

# Load ทีละ chunk (สำหรับไฟล์ใหญ่มาก)
chunks = []
for chunk in pd.read_csv('huge_file.csv', chunksize=100000):
    # Reduce memory per chunk
    chunk = reduce_mem_usage(chunk, verbose=False)
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"Final memory: {memory_usage(df)}")

# บันทึกเป็น Parquet (เล็กกว่า CSV มาก)
save_data(df, 'processed.parquet')
```

### Example 3: Benchmark Different Formats

```python
from kaggle_utils import Timer, save_data, load_data
import pandas as pd

# สร้าง test data
df = pd.DataFrame(np.random.randn(1000000, 50))

formats = ['csv', 'parquet', 'feather', 'pickle']
results = []

for fmt in formats:
    filename = f'test.{fmt}'
    
    # Test write speed
    with Timer(f"Write {fmt}") as t_write:
        save_data(df, filename)
    
    # Test read speed
    with Timer(f"Read {fmt}") as t_read:
        df_loaded = load_data(filename)
    
    # File size
    size_mb = os.path.getsize(filename) / 1024**2
    
    results.append({
        'Format': fmt,
        'Write (s)': t_write.elapsed,
        'Read (s)': t_read.elapsed,
        'Size (MB)': size_mb
    })

print(pd.DataFrame(results))
```

---

## Tips & Best Practices

### 1. ใช้ Parquet แทน CSV

```python
# ❌ Slow
df = pd.read_csv('large_file.csv')  # 30 seconds
df.to_csv('output.csv')  # 45 seconds

# ✅ Fast
df = pd.read_parquet('large_file.parquet')  # 2 seconds
df.to_parquet('output.parquet')  # 3 seconds
```

**ข้อดี Parquet:**
- เร็วกว่า CSV 10-100x
- เล็กกว่า CSV 3-10x
- เก็บ data types อัตโนมัติ
- รองรับ compression

### 2. Reduce Memory เสมอ

```python
# ลด memory ทันทีหลัง load
train = load_data('train.csv')
train = reduce_mem_usage(train)

test = load_data('test.csv')
test = reduce_mem_usage(test)

# จะประหยัด RAM 50-75%!
```

### 3. Set Seed สำหรับ Reproducibility

```python
from kaggle_utils import set_seed

# ตั้งค่า seed ตั้งแต่เริ่ม notebook
set_seed(42)

# ทุก random operation จะได้ผลเหมือนเดิม
```

### 4. ใช้ Timer เพื่อ Optimize

```python
from kaggle_utils import Timer

# หา bottleneck
with Timer("Feature engineering"):
    X = create_features(df)  # 45s

with Timer("Training"):
    model.fit(X, y)  # 120s

# → ควร optimize training!
```

### 5. Check Memory ก่อน Load

```python
from kaggle_utils import check_environment, estimate_file_size

# Check available RAM
env = check_environment()
ram_available = env['ram_total_gb']

# Estimate file size
file_size_gb = os.path.getsize('huge.csv') / 1024**3

if file_size_gb > ram_available * 0.5:
    print("⚠️  File too large! Use chunking:")
    # Load ทีละ chunk
    chunks = pd.read_csv('huge.csv', chunksize=100000)
else:
    df = load_data('huge.csv')
```

---

## Common Issues

### Issue 1: Kaggle API ไม่ทำงาน

```python
# Error: 401 Unauthorized

# Fix:
# 1. ตรวจสอบว่า kaggle.json ถูกต้อง
# 2. ดาวน์โหลด kaggle.json ใหม่จาก kaggle.com
# 3. Run setup_kaggle() อีกครั้ง

setup_kaggle(kaggle_json_path='/path/to/kaggle.json')
```

### Issue 2: Out of Memory

```python
# Error: MemoryError

# Fix 1: Reduce memory
df = reduce_mem_usage(df)

# Fix 2: Load ทีละ chunk
chunks = []
for chunk in pd.read_csv('file.csv', chunksize=50000):
    chunk = reduce_mem_usage(chunk, verbose=False)
    chunks.append(chunk)
df = pd.concat(chunks)

# Fix 3: ใช้ fewer columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2', 'col3'])
```

### Issue 3: Encoding Error

```python
# Error: UnicodeDecodeError

# Fix: ใช้ encoding='auto'
df = load_data('file.csv', encoding='auto')

# หรือลองหลาย encodings
for enc in ['utf-8', 'latin1', 'cp1252']:
    try:
        df = pd.read_csv('file.csv', encoding=enc)
        print(f"✅ Success with {enc}")
        break
    except:
        continue
```

---

## สรุป Functions

### Environment
- `setup_colab()` - Setup Google Colab
- `setup_kaggle()` - Setup Kaggle API
- `check_environment()` - ตรวจสอบ environment

### Data I/O
- `load_data()` - โหลดข้อมูล (CSV, Parquet, etc.)
- `save_data()` - บันทึกข้อมูล
- `download_kaggle_dataset()` - ดาวน์โหลดจาก Kaggle

### Submission
- `create_submission()` - สร้างไฟล์ submission

### Timing
- `timer()` - Decorator วัดเวลา
- `Timer` - Class วัดเวลา

### Memory
- `reduce_mem_usage()` - ลด memory usage
- `memory_usage()` - ดู memory usage
- `estimate_file_size()` - ประมาณขนาดไฟล์

### Others
- `set_seed()` - Set random seed
- `notify()` - ส่ง notification
- `quick_info()` - ดูข้อมูลสรุป (from preprocessing)

---

## References

- [Pandas I/O Documentation](https://pandas.pydata.org/docs/user_guide/io.html)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Parquet Format](https://parquet.apache.org/)
