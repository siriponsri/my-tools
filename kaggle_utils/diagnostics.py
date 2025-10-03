"""
Data Validation and Interpretation Tools
สำหรับตรวจสอบคุณภาพข้อมูล ตรวจจับปัญหา และให้คำแนะนำ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA QUALITY VALIDATION ====================

def check_data_quality(df, target_col=None, show_details=True):
    """
    ตรวจสอบคุณภาพข้อมูลโดยรวม
    คืนค่า: dictionary ของปัญหาที่พบและคำแนะนำ
    """
    print("="*70)
    print("🔍 DATA QUALITY CHECK")
    print("="*70)
    
    issues = []
    warnings_list = []
    suggestions = []
    
    # 1. Missing Values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    high_missing = missing_pct[missing_pct > 50]
    
    if len(high_missing) > 0:
        issues.append(f"⚠️  {len(high_missing)} features มี missing values > 50%")
        suggestions.append(f"💡 พิจารณาลบ features: {list(high_missing.index)}")
        if show_details:
            print("\n📊 Features with high missing values (>50%):")
            print(high_missing.sort_values(ascending=False))
    
    # 2. Constant Features
    constant_features = []
    for col in df.columns:
        if df[col].dtype != 'object':
            if df[col].nunique() == 1:
                constant_features.append(col)
    
    if constant_features:
        issues.append(f"⚠️  {len(constant_features)} features มีค่าคงที่ (ไม่มีความแปรปรวน)")
        suggestions.append(f"💡 ควรลบ constant features: {constant_features}")
        if show_details:
            print(f"\n⚠️  Constant features: {constant_features}")
    
    # 3. Duplicate Features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        duplicate_features = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        if duplicate_features:
            issues.append(f"⚠️  {len(duplicate_features)} features มีความสัมพันธ์สูงมาก (>0.95)")
            suggestions.append(f"💡 พิจารณาลบ features ที่ซ้ำซ้อน: {duplicate_features[:5]}")
            if show_details:
                print(f"\n⚠️  Highly correlated features: {duplicate_features[:10]}")
    
    # 4. Cardinality Check (สำหรับ categorical)
    cat_cols = df.select_dtypes(include=['object']).columns
    high_cardinality = []
    for col in cat_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:
            high_cardinality.append((col, unique_ratio))
    
    if high_cardinality:
        warnings_list.append(f"⚠️  {len(high_cardinality)} categorical features มี cardinality สูง")
        suggestions.append("💡 พิจารณาใช้ target encoding หรือ feature hashing")
        if show_details:
            print(f"\n⚠️  High cardinality features:")
            for col, ratio in high_cardinality[:5]:
                print(f"   - {col}: {ratio:.1%} unique values")
    
    # 5. Target Distribution (ถ้ามี target_col)
    if target_col and target_col in df.columns:
        print(f"\n📊 Target Distribution ({target_col}):")
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
            # Classification
            print(df[target_col].value_counts())
            class_balance = df[target_col].value_counts(normalize=True)
            if class_balance.min() < 0.1:
                issues.append("⚠️  Class Imbalance ตรวจพบ!")
                suggestions.append("💡 พิจารณาใช้ SMOTE, class_weight, หรือ stratified sampling")
        else:
            # Regression
            print(f"Mean: {df[target_col].mean():.2f}")
            print(f"Std: {df[target_col].std():.2f}")
            print(f"Min: {df[target_col].min():.2f}")
            print(f"Max: {df[target_col].max():.2f}")
            
            # Check skewness
            skewness = df[target_col].skew()
            if abs(skewness) > 1:
                warnings_list.append(f"⚠️  Target มี skewness สูง ({skewness:.2f})")
                suggestions.append("💡 พิจารณา log transform สำหรับ target")
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if not issues and not warnings_list:
        print("✅ ไม่พบปัญหาสำคัญ! ข้อมูลมีคุณภาพดี")
    else:
        print(f"\n❌ Issues found: {len(issues)}")
        for issue in issues:
            print(f"   {issue}")
        
        if warnings_list:
            print(f"\n⚠️  Warnings: {len(warnings_list)}")
            for warning in warnings_list:
                print(f"   {warning}")
    
    if suggestions:
        print(f"\n💡 Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    
    return {
        'issues': issues,
        'warnings': warnings_list,
        'suggestions': suggestions,
        'constant_features': constant_features,
        'high_missing_features': list(high_missing.index) if len(high_missing) > 0 else [],
        'duplicate_features': duplicate_features if 'duplicate_features' in locals() else []
    }

# ==================== DATA LEAKAGE DETECTION ====================

def detect_leakage(train_df, target_col, test_df=None, threshold=0.95):
    """
    ตรวจจับ Data Leakage โดยหา features ที่มีความสัมพันธ์สูงผิดปกติกับ target
    """
    print("="*70)
    print("🔍 DATA LEAKAGE DETECTION")
    print("="*70)
    
    suspicious_features = []
    
    # 1. Perfect Correlation Check
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        correlations = train_df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        high_corr = correlations[correlations.abs() > threshold]
        
        if len(high_corr) > 0:
            print(f"\n⚠️  พบ {len(high_corr)} features ที่มีความสัมพันธ์สูงผิดปกติกับ target (>{threshold}):")
            for feat, corr in high_corr.items():
                print(f"   - {feat}: {corr:.4f}")
                suspicious_features.append(feat)
            print("\n❌ อาจเป็น Data Leakage! ตรวจสอบ features เหล่านี้")
    
    # 2. Mutual Information Check (สูงผิดปกติ)
    if train_df[target_col].dtype != 'object' and train_df[target_col].nunique() > 20:
        # Regression
        mi_scores = mutual_info_regression(train_df[numeric_cols], train_df[target_col])
    else:
        # Classification
        mi_scores = mutual_info_classif(train_df[numeric_cols], train_df[target_col])
    
    mi_df = pd.DataFrame({'feature': numeric_cols, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # Features ที่มี MI สูงกว่าค่าเฉลี่ย 3 เท่า
    mean_mi = mi_df['mi_score'].mean()
    suspicious_mi = mi_df[mi_df['mi_score'] > 3 * mean_mi]
    
    if len(suspicious_mi) > 0:
        print(f"\n⚠️  Features ที่มี Mutual Information สูงผิดปกติ:")
        print(suspicious_mi.head(10))
    
    # 3. Train-Test Distribution Check
    if test_df is not None:
        print("\n🔍 Checking Train-Test Distribution Differences...")
        from scipy.stats import ks_2samp
        
        significant_diff = []
        for col in numeric_cols:
            if col in test_df.columns:
                stat, pvalue = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
                if pvalue < 0.01:  # Significant difference
                    significant_diff.append((col, pvalue))
        
        if significant_diff:
            print(f"\n⚠️  {len(significant_diff)} features มี distribution แตกต่างกันระหว่าง train-test:")
            for col, pval in significant_diff[:5]:
                print(f"   - {col}: p-value = {pval:.6f}")
            print("\n💡 อาจมี data leakage หรือ train/test มาจากคนละ distribution")
    
    # 4. Temporal Leakage Check (ถ้ามี date columns)
    date_cols = train_df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        print(f"\n📅 Date columns found: {list(date_cols)}")
        print("⚠️  Warning: ระวัง temporal leakage!")
        print("💡 ตรวจสอบว่า features ถูกสร้างก่อนหรือหลัง target เกิดขึ้น")
    
    print("\n" + "="*70)
    if suspicious_features:
        print(f"❌ พบ {len(suspicious_features)} features ที่น่าสงสัย")
        print("💡 แนะนำ: ตรวจสอบว่า features เหล่านี้ถูกสร้างอย่างไร และควรอยู่ใน model หรือไม่")
    else:
        print("✅ ไม่พบสัญญาณชัดเจนของ data leakage")
    
    return suspicious_features

# ==================== MULTICOLLINEARITY DETECTION ====================

def check_multicollinearity(df, threshold=0.8):
    """
    ตรวจสอบ Multicollinearity (features ที่มีความสัมพันธ์กันสูง)
    """
    print("="*70)
    print("🔍 MULTICOLLINEARITY CHECK")
    print("="*70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("⚠️  ต้องมี numeric features อย่างน้อย 2 features")
        return []
    
    corr_matrix = df[numeric_cols].corr().abs()
    
    # หา feature pairs ที่มีความสัมพันธ์สูง
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\n⚠️  พบ {len(high_corr_pairs)} feature pairs ที่มีความสัมพันธ์สูง (>{threshold}):\n")
        pairs_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
        print(pairs_df.head(10))
        
        print(f"\n💡 คำแนะนำ:")
        print(f"   1. พิจารณาลบ features ที่ซ้ำซ้อน")
        print(f"   2. ใช้ PCA หรือ feature selection")
        print(f"   3. ใช้ regularization (Ridge/Lasso)")
        
        # แนะนำ features ที่ควรลบ
        feature_counts = {}
        for pair in high_corr_pairs:
            feature_counts[pair['feature_1']] = feature_counts.get(pair['feature_1'], 0) + 1
            feature_counts[pair['feature_2']] = feature_counts.get(pair['feature_2'], 0) + 1
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🎯 Features ที่มีความสัมพันธ์กับ features อื่นมากที่สุด:")
        for feat, count in sorted_features[:5]:
            print(f"   - {feat}: มีความสัมพันธ์กับ {count} features")
    else:
        print("✅ ไม่พบปัญหา multicollinearity")
    
    return high_corr_pairs

# ==================== MODEL RECOMMENDATION ====================

def suggest_models(train_df, target_col, task='auto'):
    """
    แนะนำโมเดลที่เหมาะสมตามลักษณะข้อมูล
    """
    print("="*70)
    print("🤖 MODEL RECOMMENDATION")
    print("="*70)
    
    n_samples = len(train_df)
    n_features = len(train_df.columns) - 1
    
    # Detect task type
    if task == 'auto':
        if train_df[target_col].dtype == 'object' or train_df[target_col].nunique() < 20:
            task = 'classification'
        else:
            task = 'regression'
    
    print(f"\n📊 Dataset Info:")
    print(f"   - Task: {task.upper()}")
    print(f"   - Samples: {n_samples:,}")
    print(f"   - Features: {n_features}")
    print(f"   - Sample/Feature Ratio: {n_samples/n_features:.1f}")
    
    recommendations = []
    
    # Size-based recommendations
    if n_samples < 1000:
        print(f"\n📉 Dataset Size: SMALL (<1,000 samples)")
        recommendations.extend([
            "🥇 Ridge/Lasso Regression (simple, less overfitting)" if task == 'regression' else "🥇 Logistic Regression",
            "🥈 Random Forest (small n_estimators)",
            "🥉 Gradient Boosting (small n_estimators, high learning_rate)"
        ])
        print("⚠️  Warning: ระวัง overfitting! ใช้ regularization และ cross-validation")
        
    elif n_samples < 10000:
        print(f"\n📊 Dataset Size: MEDIUM (1K-10K samples)")
        recommendations.extend([
            "🥇 LightGBM / XGBoost",
            "🥈 Random Forest",
            "🥉 CatBoost (ถ้ามี categorical features เยอะ)"
        ])
        
    else:
        print(f"\n📈 Dataset Size: LARGE (>10K samples)")
        recommendations.extend([
            "🥇 LightGBM (เร็ว, แม่นยำ)",
            "🥈 XGBoost",
            "🥉 Neural Networks (ถ้ามี features เยอะมาก)"
        ])
    
    # Feature-based recommendations
    cat_features = len(train_df.select_dtypes(include=['object']).columns)
    if cat_features > 5:
        print(f"\n📝 Categorical Features: {cat_features}")
        recommendations.append("💡 CatBoost เหมาะมาก (handle categorical โดยตรง)")
    
    # Ratio-based warnings
    if n_samples / n_features < 5:
        print(f"\n⚠️  Warning: Feature มากกว่า samples!")
        print("💡 แนะนำ: Feature selection, PCA, หรือ regularization")
    
    print(f"\n🎯 Recommended Models:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec}")
    
    return recommendations

# ==================== OVERFITTING DETECTION ====================

def detect_overfitting(model, X_train, y_train, X_val, y_val, task='regression'):
    """
    ตรวจจับ Overfitting โดยเปรียบเทียบ train vs validation score
    """
    print("="*70)
    print("🔍 OVERFITTING DETECTION")
    print("="*70)
    
    # Train and predict
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    if task == 'regression':
        train_score = np.sqrt(mean_squared_error(y_train, train_pred))
        val_score = np.sqrt(mean_squared_error(y_val, val_pred))
        metric_name = "RMSE"
        # Lower is better for RMSE
        gap = val_score - train_score
        gap_pct = (gap / train_score) * 100
    else:
        if hasattr(model, 'predict_proba'):
            train_pred_proba = model.predict_proba(X_train)[:, 1]
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            train_score = roc_auc_score(y_train, train_pred_proba)
            val_score = roc_auc_score(y_val, val_pred_proba)
        else:
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
        metric_name = "AUC" if hasattr(model, 'predict_proba') else "Accuracy"
        # Higher is better for AUC/Accuracy
        gap = train_score - val_score
        gap_pct = (gap / train_score) * 100
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Train {metric_name}: {train_score:.4f}")
    print(f"   Val {metric_name}: {val_score:.4f}")
    print(f"   Gap: {gap:.4f} ({gap_pct:.1f}%)")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if abs(gap_pct) < 5:
        print("   ✅ Good fit! Train และ validation scores ใกล้เคียงกัน")
        status = "good_fit"
    elif gap_pct > 15:
        print("   ❌ OVERFITTING detected!")
        print("   💡 คำแนะนำ:")
        print("      1. เพิ่มข้อมูล training")
        print("      2. ใช้ regularization (Ridge, Lasso, or L1/L2)")
        print("      3. ลด model complexity (max_depth, n_estimators)")
        print("      4. เพิ่ม dropout (สำหรับ neural networks)")
        print("      5. ใช้ early stopping")
        print("      6. ทำ feature selection")
        status = "overfitting"
    elif gap_pct < -15:
        print("   ⚠️  UNDERFITTING detected!")
        print("   💡 คำแนะนำ:")
        print("      1. เพิ่ม model complexity")
        print("      2. เพิ่ม features")
        print("      3. ลด regularization")
        print("      4. Train นานขึ้น (เพิ่ม n_estimators)")
        status = "underfitting"
    else:
        print("   ⚠️  Slight overfitting - ยังอยู่ในเกณฑ์ที่ยอมรับได้")
        status = "slight_overfitting"
    
    return {
        'train_score': train_score,
        'val_score': val_score,
        'gap': gap,
        'gap_percentage': gap_pct,
        'status': status
    }

# ==================== LEARNING CURVE ANALYSIS ====================

def plot_learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
    """
    Plot learning curve เพื่อวิเคราะห์ overfitting/underfitting
    """
    print("📈 Generating learning curve...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring,
        n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = -train_scores.mean(axis=1) if 'neg' in scoring else train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1) if 'neg' in scoring else val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Diagnosis
    final_gap = abs(val_mean[-1] - train_mean[-1])
    if final_gap < 0.05 * train_mean[-1]:
        diagnosis = "✅ Good fit"
    elif val_mean[-1] < train_mean[-1]:
        diagnosis = "❌ Overfitting - validation score ต่ำกว่า train score"
    else:
        diagnosis = "⚠️  Underfitting - ทั้ง train และ val scores ต่ำ"
    
    plt.text(0.02, 0.98, diagnosis, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔍 Analysis:")
    print(f"   Final Training Score: {train_mean[-1]:.4f}")
    print(f"   Final Validation Score: {val_mean[-1]:.4f}")
    print(f"   {diagnosis}")

# ==================== COMPREHENSIVE DIAGNOSIS ====================

def quick_diagnosis(train_df, target_col, test_df=None, model=None, X_val=None, y_val=None):
    """
    วินิจฉัยด่วน - รวมการตรวจสอบทั้งหมดในฟังก์ชันเดียว
    """
    print("🏥 QUICK DIAGNOSIS - Comprehensive Data & Model Check")
    print("="*70)
    
    # 1. Data Quality
    print("\n" + "🔹"*35)
    quality_report = check_data_quality(train_df, target_col, show_details=False)
    
    # 2. Data Leakage
    print("\n" + "🔹"*35)
    suspicious_features = detect_leakage(train_df, target_col, test_df)
    
    # 3. Multicollinearity
    print("\n" + "🔹"*35)
    X = train_df.drop(columns=[target_col])
    multicollinearity = check_multicollinearity(X, threshold=0.8)
    
    # 4. Model Recommendation
    print("\n" + "🔹"*35)
    model_recommendations = suggest_models(train_df, target_col)
    
    # 5. Overfitting Check (ถ้ามี model)
    if model is not None and X_val is not None and y_val is not None:
        print("\n" + "🔹"*35)
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        task = 'classification' if train_df[target_col].dtype == 'object' or train_df[target_col].nunique() < 20 else 'regression'
        overfitting_report = detect_overfitting(model, X_train, y_train, X_val, y_val, task)
    
    # 6. Summary Report
    print("\n" + "="*70)
    print("📋 FINAL REPORT")
    print("="*70)
    
    total_issues = len(quality_report['issues']) + len(quality_report['warnings']) + len(suspicious_features)
    
    if total_issues == 0:
        print("\n✅✅✅ ข้อมูลมีคุณภาพดี! ไม่พบปัญหาสำคัญ")
    else:
        print(f"\n⚠️  พบปัญหาทั้งหมด: {total_issues}")
        
        print(f"\n🔴 Priority Issues:")
        priority_actions = []
        
        if quality_report['constant_features']:
            priority_actions.append("1. ลบ constant features")
        if suspicious_features:
            priority_actions.append("2. ตรวจสอบ data leakage features")
        if len(multicollinearity) > 10:
            priority_actions.append("3. จัดการ multicollinearity")
        if quality_report['high_missing_features']:
            priority_actions.append("4. จัดการ missing values")
        
        for action in priority_actions:
            print(f"   {action}")
    
    print(f"\n🎯 Recommended Next Steps:")
    print(f"   1. {model_recommendations[0] if model_recommendations else 'Run quick_model_comparison()'}")
    print(f"   2. ใช้ cross-validation เพื่อประเมินโมเดล")
    print(f"   3. Plot learning curves เพื่อตรวจสอบ overfitting")
    print(f"   4. ทำ feature engineering เพิ่มเติม")
    
    return {
        'quality_report': quality_report,
        'suspicious_features': suspicious_features,
        'multicollinearity_count': len(multicollinearity),
        'model_recommendations': model_recommendations
    }

# ==================== FEATURE IMPORTANCE ANALYSIS ====================

def analyze_feature_importance_detailed(model, X, y, feature_names, top_n=20):
    """
    วิเคราะห์ความสำคัญของ features แบบละเอียด
    """
    print("="*70)
    print("🎯 DETAILED FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    from sklearn.inspection import permutation_importance
    
    # 1. Built-in importance (ถ้ามี)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Built-in Feature Importance (Top 20):")
        print(importance_df.head(top_n))
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    # 2. Permutation Importance
    print("\n🔄 Computing Permutation Importance...")
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n📊 Permutation Importance (Top 20):")
    print(perm_importance_df.head(top_n))
    
    # 3. Features ที่ควรลบ
    low_importance = importance_df[importance_df['importance'] < 0.001] if hasattr(model, 'feature_importances_') else perm_importance_df[perm_importance_df['importance_mean'] < 0.001]
    
    if len(low_importance) > 0:
        print(f"\n💡 พบ {len(low_importance)} features ที่มี importance ต่ำมาก (<0.001):")
        print(f"   พิจารณาลบ features: {list(low_importance['feature'].head(10))}")
    
    return importance_df if hasattr(model, 'feature_importances_') else perm_importance_df

print("✅ Diagnostics module loaded successfully!")