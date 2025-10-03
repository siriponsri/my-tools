"""
kaggle-utils: ชุดเครื่องมือสำหรับแข่งขัน Kaggle

เหมาะสำหรับ:
- มือใหม่ที่เริ่มแข่ง Kaggle
- คนที่ต้องการทำ baseline ไว
- คนที่ไม่อยากเขียนโค้ดซ้ำๆ

Modules:
- preprocessing: Data preprocessing & feature engineering
- models: Model wrappers with CV built-in
- ensemble: Ensemble methods
- outliers: Outlier detection & handling
- hyperparams: Hyperparameter tuning
- visualization: Plotting functions
- metrics: Evaluation metrics
- diagnostics: Data quality & model diagnostics
- utils: Utility functions
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# ============================================================
# Import main functions from each module
# ============================================================

# Preprocessing
from .preprocessing import (
    quick_info,
    reduce_mem_usage,
    handle_missing_values,
    create_missing_indicators,
    encode_categorical,
    target_encode,
    scale_features,
    create_time_features,
    create_polynomial_features,
    create_interaction_features,
    create_aggregation_features,
    create_ratio_features,
    create_bins,
    create_text_features,
    auto_feature_selection,
    remove_low_variance_features,
    split_train_test_by_date,
)

# Models
from .models import (
    SKLearnWrapper,
    RandomForestWrapper,
    RidgeWrapper,
    LassoWrapper,
    ElasticNetWrapper,
    LGBWrapper,
    XGBWrapper,
    CatBoostWrapper,
    quick_model_comparison,
    quick_classification_comparison,
    compare_scalers,
    create_pipeline,
)

# Ensemble
from .ensemble import (
    WeightedEnsemble,
    StackingEnsemble,
    DynamicEnsemble,
    create_voting_ensemble,
    blend_predictions,
    optimize_blend_weights,
)

# Outliers
from .outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    detect_outliers_lof,
    detect_outliers_elliptic,
    detect_outliers_ensemble,
    handle_outliers,
    plot_outliers,
    plot_outliers_comparison,
    outlier_summary,
)

# Hyperparameters
from .hyperparams import (
    tune_hyperparameters,
    suggest_params_lightgbm,
    suggest_params_xgboost,
    suggest_params_catboost,
    suggest_params_random_forest,
    grid_search_cv,
    random_search_cv,
    bayesian_optimization,
)

# Visualization
from .visualization import (
    plot_feature_importance,
    plot_feature_importance_comparison,
    plot_distributions,
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_correlation_with_target,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_predictions,
    plot_missing_values,
)

# Metrics
from .metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    rmse,
    mae,
    mape,
    rmsle,
    r2_score_custom,
    classification_report_custom,
    confusion_matrix_metrics,
    optimal_threshold,
    kaggle_metric,
)

# Diagnostics
from .diagnostics import (
    check_data_quality,
    detect_leakage,
    check_multicollinearity,
    suggest_models,
    detect_overfitting,
    plot_learning_curve,
    quick_diagnosis,
    analyze_feature_importance_detailed,
)

# Utils
from .utils import (
    setup_colab,
    setup_kaggle,
    download_kaggle_dataset,
    load_data,
    save_data,
    create_submission,
    set_seed,
    timer,
    Timer,
    notify,
    check_environment,
    memory_usage,
    estimate_file_size,
)


# ============================================================
# Define __all__ for "from kaggle_utils import *"
# ============================================================

__all__ = [
    # Version
    '__version__',
    
    # Preprocessing
    'quick_info',
    'reduce_mem_usage',
    'handle_missing_values',
    'create_missing_indicators',
    'encode_categorical',
    'target_encode',
    'scale_features',
    'create_time_features',
    'create_polynomial_features',
    'create_interaction_features',
    'create_aggregation_features',
    'create_ratio_features',
    'create_bins',
    'create_text_features',
    'auto_feature_selection',
    'remove_low_variance_features',
    'split_train_test_by_date',
    
    # Models
    'SKLearnWrapper',
    'RandomForestWrapper',
    'RidgeWrapper',
    'LassoWrapper',
    'ElasticNetWrapper',
    'LGBWrapper',
    'XGBWrapper',
    'CatBoostWrapper',
    'quick_model_comparison',
    'quick_classification_comparison',
    'compare_scalers',
    'create_pipeline',
    
    # Ensemble
    'WeightedEnsemble',
    'StackingEnsemble',
    'DynamicEnsemble',
    'create_voting_ensemble',
    'blend_predictions',
    'optimize_blend_weights',
    
    # Outliers
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'detect_outliers_isolation_forest',
    'detect_outliers_lof',
    'detect_outliers_elliptic',
    'detect_outliers_ensemble',
    'handle_outliers',
    'plot_outliers',
    'plot_outliers_comparison',
    'outlier_summary',
    
    # Hyperparameters
    'tune_hyperparameters',
    'suggest_params_lightgbm',
    'suggest_params_xgboost',
    'suggest_params_catboost',
    'suggest_params_random_forest',
    'grid_search_cv',
    'random_search_cv',
    'bayesian_optimization',
    
    # Visualization
    'plot_feature_importance',
    'plot_feature_importance_comparison',
    'plot_distributions',
    'plot_target_distribution',
    'plot_correlation_heatmap',
    'plot_correlation_with_target',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_predictions',
    'plot_missing_values',
    
    # Metrics
    'calculate_regression_metrics',
    'calculate_classification_metrics',
    'rmse',
    'mae',
    'mape',
    'rmsle',
    'r2_score_custom',
    'classification_report_custom',
    'confusion_matrix_metrics',
    'optimal_threshold',
    'kaggle_metric',
    
    # Diagnostics
    'check_data_quality',
    'detect_leakage',
    'check_multicollinearity',
    'suggest_models',
    'detect_overfitting',
    'plot_learning_curve',
    'quick_diagnosis',
    'analyze_feature_importance_detailed',
    
    # Utils
    'setup_colab',
    'setup_kaggle',
    'download_kaggle_dataset',
    'load_data',
    'save_data',
    'create_submission',
    'set_seed',
    'timer',
    'Timer',
    'notify',
    'check_environment',
    'memory_usage',
    'estimate_file_size',
]


# ============================================================
# Package-level convenience functions
# ============================================================

def print_version():
    """แสดง version ของ package"""
    print(f"kaggle-utils version {__version__}")


def list_modules():
    """แสดง modules ทั้งหมดที่มี"""
    modules = [
        ('preprocessing', 'Data preprocessing & feature engineering'),
        ('models', 'Model wrappers with CV built-in'),
        ('ensemble', 'Ensemble methods'),
        ('outliers', 'Outlier detection & handling'),
        ('hyperparams', 'Hyperparameter tuning'),
        ('visualization', 'Plotting functions'),
        ('metrics', 'Evaluation metrics'),
        ('diagnostics', 'Data quality & model diagnostics'),
        ('utils', 'Utility functions'),
    ]
    
    print("=" * 70)
    print("kaggle-utils Modules")
    print("=" * 70)
    for name, desc in modules:
        print(f"  {name:15s}: {desc}")
    print("=" * 70)


def quick_start_guide():
    """แสดงคู่มือการใช้งานแบบย่อ"""
    guide = """
    ============================================================
    Quick Start Guide - kaggle-utils
    ============================================================
    
    1. Setup (ถ้าใช้ Colab):
       >>> from kaggle_utils import setup_colab
       >>> setup_colab()
    
    2. Load & Inspect Data:
       >>> from kaggle_utils import load_data, quick_info, reduce_mem_usage
       >>> train = load_data('train.csv')
       >>> quick_info(train)
       >>> train = reduce_mem_usage(train)
    
    3. Diagnose Problems:
       >>> from kaggle_utils import quick_diagnosis
       >>> report = quick_diagnosis(train, target_col='target')
    
    4. Preprocess:
       >>> from kaggle_utils import handle_missing_values, encode_categorical
       >>> train = handle_missing_values(train, strategy='auto')
       >>> train = encode_categorical(train, method='label')
    
    5. Train Models:
       >>> from kaggle_utils.models import LGBWrapper
       >>> lgb = LGBWrapper(n_splits=5)
       >>> lgb.train(X_train, y_train, X_test=X_test)
    
    6. Ensemble:
       >>> from kaggle_utils import blend_predictions
       >>> final = blend_predictions([pred1, pred2, pred3], weights=[0.5, 0.3, 0.2])
    
    7. Submit:
       >>> from kaggle_utils import create_submission
       >>> create_submission(test['id'], final, 'submission.csv')
    
    ============================================================
    
    For detailed guides, check:
    - preprocessing_guide.md
    - models_guide.md
    - ensemble_guide.md
    - diagnostics_guide.md
    - visualization_guide.md
    
    ============================================================
    """
    print(guide)


# ============================================================
# Initialize
# ============================================================

# Suppress warnings by default
import warnings
warnings.filterwarnings('ignore')

# Set display options for pandas
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)