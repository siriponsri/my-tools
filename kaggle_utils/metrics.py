"""
📊 Metrics Module - Evaluation Metrics for Kaggle Competitions

Functions:
- calculate_regression_metrics() - คำนวณ metrics สำหรับ regression
- calculate_classification_metrics() - คำนวณ metrics สำหรับ classification
- rmse() - Root Mean Squared Error
- mae() - Mean Absolute Error
- mape() - Mean Absolute Percentage Error
- r2_score_custom() - R² Score
- rmsle() - Root Mean Squared Log Error
- classification_report_custom() - สร้าง classification report พร้อมคำอธิบาย
- confusion_matrix_metrics() - คำนวณ metrics จาก confusion matrix
- optimal_threshold() - หา threshold ที่ดีที่สุด
- kaggle_metric() - ใช้ metric ตามที่ Kaggle กำหนด
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAE score
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAPE score (as percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Log Error
    
    Parameters:
    -----------
    y_true : array-like
        True values (must be positive)
    y_pred : array-like
        Predicted values (must be positive)
    
    Returns:
    --------
    float
        RMSLE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip negative predictions to small positive value
    y_pred = np.maximum(y_pred, 1e-10)
    
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def r2_score_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² Score (Coefficient of Determination)
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        R² score
    """
    return r2_score(y_true, y_pred)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    คำนวณ regression metrics ทั้งหมด
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    verbose : bool, default=True
        แสดงผลลัพธ์
    
    Returns:
    --------
    dict
        Dictionary ของ metrics
    """
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R2': r2_score_custom(y_true, y_pred),
    }
    
    # MAPE - ถ้าไม่มี 0 ใน y_true
    if np.all(y_true != 0):
        metrics['MAPE'] = mape(y_true, y_pred)
    
    # RMSLE - ถ้าทุกค่าเป็น positive
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        metrics['RMSLE'] = rmsle(y_true, y_pred)
    
    if verbose:
        print("=" * 60)
        print("📊 Regression Metrics")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"{metric_name:10s}: {value:.6f}")
        print("=" * 60)
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'binary',
    verbose: bool = True
) -> Dict[str, float]:
    """
    คำนวณ classification metrics ทั้งหมด
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (สำหรับ AUC และ Log Loss)
    average : str, default='binary'
        'binary', 'micro', 'macro', 'weighted'
    verbose : bool, default=True
        แสดงผลลัพธ์
    
    Returns:
    --------
    dict
        Dictionary ของ metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # AUC และ Log Loss - ต้องมี probabilities
    if y_pred_proba is not None:
        try:
            if average == 'binary':
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                              average=average, multi_class='ovr')
            metrics['Log Loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            pass  # Skip ถ้าคำนวณไม่ได้
    
    if verbose:
        print("=" * 60)
        print("📊 Classification Metrics")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s}: {value:.6f}")
        print("=" * 60)
    
    return metrics


def classification_report_custom(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    สร้าง classification report แบบละเอียด
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list, optional
        ชื่อของแต่ละ class
    verbose : bool, default=True
        แสดงผลลัพธ์
    
    Returns:
    --------
    pd.DataFrame
        Classification report
    """
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    
    if verbose:
        print("=" * 60)
        print("📋 Classification Report")
        print("=" * 60)
        print(report_df.to_string())
        print("=" * 60)
    
    return report_df


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, float]]:
    """
    คำนวณ metrics จาก confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    verbose : bool, default=True
        แสดงผลลัพธ์
    
    Returns:
    --------
    dict
        Dictionary มี confusion matrix และ metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    result = {
        'confusion_matrix': cm,
    }
    
    # สำหรับ binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        result.update({
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp,
            'Sensitivity/Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        })
    
    if verbose:
        print("=" * 60)
        print("🔢 Confusion Matrix Metrics")
        print("=" * 60)
        print("Confusion Matrix:")
        print(cm)
        print()
        for key, value in result.items():
            if key != 'confusion_matrix':
                print(f"{key:20s}: {value:.4f}")
        print("=" * 60)
    
    return result


def optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'f1',
    verbose: bool = True
) -> Tuple[float, float]:
    """
    หา threshold ที่ดีที่สุดสำหรับ binary classification
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    metric : str, default='f1'
        'f1', 'precision', 'recall', 'accuracy'
    verbose : bool, default=True
        แสดงผลลัพธ์
    
    Returns:
    --------
    tuple
        (best_threshold, best_score)
    """
    thresholds = np.arange(0.1, 1.0, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    if verbose:
        print("=" * 60)
        print(f"🎯 Optimal Threshold ({metric.upper()})")
        print("=" * 60)
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Best {metric.upper()}: {best_score:.6f}")
        print("=" * 60)
    
    return best_threshold, best_score


def kaggle_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    **kwargs
) -> float:
    """
    คำนวณ metric ตามที่ Kaggle competitions ใช้
    
    Parameters:
    -----------
    y_true : array-like
        True values/labels
    y_pred : array-like
        Predicted values/labels or probabilities
    metric : str
        Metric name: 'rmse', 'mae', 'rmsle', 'auc', 'logloss', 'accuracy', 'f1'
    **kwargs : dict
        Additional arguments for specific metrics
    
    Returns:
    --------
    float
        Metric score
    
    Examples:
    ---------
    >>> # Regression
    >>> score = kaggle_metric(y_true, y_pred, 'rmse')
    >>> 
    >>> # Classification
    >>> score = kaggle_metric(y_true, y_pred_proba, 'auc')
    """
    metric = metric.lower()
    
    # Regression metrics
    if metric in ['rmse', 'root_mean_squared_error']:
        return rmse(y_true, y_pred)
    
    elif metric in ['mae', 'mean_absolute_error']:
        return mae(y_true, y_pred)
    
    elif metric in ['rmsle', 'root_mean_squared_log_error']:
        return rmsle(y_true, y_pred)
    
    elif metric in ['r2', 'r_squared']:
        return r2_score_custom(y_true, y_pred)
    
    elif metric == 'mape':
        return mape(y_true, y_pred)
    
    # Classification metrics
    elif metric in ['auc', 'roc_auc']:
        return roc_auc_score(y_true, y_pred, **kwargs)
    
    elif metric in ['logloss', 'log_loss']:
        return log_loss(y_true, y_pred, **kwargs)
    
    elif metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    
    elif metric == 'f1':
        return f1_score(y_true, y_pred, **kwargs)
    
    elif metric == 'precision':
        return precision_score(y_true, y_pred, **kwargs)
    
    elif metric == 'recall':
        return recall_score(y_true, y_pred, **kwargs)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Regression example
    print("\n" + "="*60)
    print("📊 REGRESSION METRICS EXAMPLE")
    print("="*60)
    
    y_true_reg = np.array([100, 150, 200, 250, 300])
    y_pred_reg = np.array([110, 145, 195, 255, 290])
    
    metrics_reg = calculate_regression_metrics(y_true_reg, y_pred_reg)
    
    # Classification example
    print("\n" + "="*60)
    print("📊 CLASSIFICATION METRICS EXAMPLE")
    print("="*60)
    
    y_true_clf = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    y_pred_clf = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.6, 0.3])
    
    metrics_clf = calculate_classification_metrics(
        y_true_clf, y_pred_clf, y_pred_proba
    )
    
    # Confusion matrix
    cm_metrics = confusion_matrix_metrics(y_true_clf, y_pred_clf)
    
    # Optimal threshold
    best_th, best_score = optimal_threshold(y_true_clf, y_pred_proba, metric='f1')