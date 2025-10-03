"""
⚙️ Hyperparameter Tuning Module

Functions:
- tune_hyperparameters() - หา hyperparameters ที่ดีที่สุด (Optuna/GridSearch/RandomSearch)
- suggest_params_lightgbm() - แนะนำ search space สำหรับ LightGBM
- suggest_params_xgboost() - แนะนำ search space สำหรับ XGBoost
- suggest_params_catboost() - แนะนำ search space สำหรับ CatBoost
- suggest_params_random_forest() - แนะนำ search space สำหรับ Random Forest
- grid_search_cv() - Grid Search with Cross Validation
- random_search_cv() - Random Search with Cross Validation
- bayesian_optimization() - Bayesian Optimization (Optuna)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

# Try import optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not installed. Install with: pip install optuna")


def suggest_params_lightgbm(
    search_space: str = 'default',
    task: str = 'regression'
) -> Dict[str, Any]:
    """
    แนะนำ hyperparameter search space สำหรับ LightGBM
    
    Parameters:
    -----------
    search_space : str, default='default'
        'narrow', 'default', 'wide'
    task : str, default='regression'
        'regression' or 'classification'
    
    Returns:
    --------
    dict
        Search space
    """
    if search_space == 'narrow':
        space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 5, 10],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    elif search_space == 'wide':
        space = {
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 500, 1000, 2000],
            'num_leaves': [15, 31, 50, 100, 200],
            'max_depth': [-1, 3, 5, 7, 10, 15],
            'min_child_samples': [5, 10, 20, 50, 100],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 1.0],
        }
    else:  # default
        space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 5, 10],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }
    
    # Add task-specific parameters
    if task == 'classification':
        space['boosting_type'] = ['gbdt', 'dart']
    
    return space


def suggest_params_xgboost(
    search_space: str = 'default',
    task: str = 'regression'
) -> Dict[str, Any]:
    """
    แนะนำ hyperparameter search space สำหรับ XGBoost
    
    Parameters:
    -----------
    search_space : str, default='default'
        'narrow', 'default', 'wide'
    task : str, default='regression'
        'regression' or 'classification'
    
    Returns:
    --------
    dict
        Search space
    """
    if search_space == 'narrow':
        space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    elif search_space == 'wide':
        space = {
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 500, 1000, 2000],
            'max_depth': [3, 5, 7, 10, 15],
            'min_child_weight': [1, 3, 5, 10],
            'gamma': [0, 0.1, 0.5, 1.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 1.0],
        }
    else:  # default
        space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }
    
    return space


def suggest_params_catboost(
    search_space: str = 'default',
    task: str = 'regression'
) -> Dict[str, Any]:
    """
    แนะนำ hyperparameter search space สำหรับ CatBoost
    
    Parameters:
    -----------
    search_space : str, default='default'
        'narrow', 'default', 'wide'
    task : str, default='regression'
        'regression' or 'classification'
    
    Returns:
    --------
    dict
        Search space
    """
    if search_space == 'narrow':
        space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 500, 1000],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
        }
    elif search_space == 'wide':
        space = {
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'iterations': [50, 100, 500, 1000, 2000],
            'depth': [3, 4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'bagging_temperature': [0, 0.5, 1.0],
            'random_strength': [0, 1, 2],
        }
    else:  # default
        space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 500, 1000],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
        }
    
    return space


def suggest_params_random_forest(
    search_space: str = 'default'
) -> Dict[str, Any]:
    """
    แนะนำ hyperparameter search space สำหรับ Random Forest
    
    Parameters:
    -----------
    search_space : str, default='default'
        'narrow', 'default', 'wide'
    
    Returns:
    --------
    dict
        Search space
    """
    if search_space == 'narrow':
        space = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    elif search_space == 'wide':
        space = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        }
    else:  # default
        space = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
    
    return space


def grid_search_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List],
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Grid Search with Cross Validation
    
    Parameters:
    -----------
    model : estimator
        Model to tune
    X : array-like
        Features
    y : array-like
        Target
    param_grid : dict
        Parameter grid
    cv : int, default=5
        Number of folds
    scoring : str, optional
        Scoring metric
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    dict
        Results with best params and score
    """
    print("=" * 60)
    print("🔍 Grid Search CV")
    print("=" * 60)
    print(f"Parameter grid: {param_grid}")
    print(f"CV folds: {cv}")
    print(f"Scoring: {scoring or 'default'}")
    print()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    grid_search.fit(X, y)
    
    print("=" * 60)
    print("✅ Grid Search Complete!")
    print("=" * 60)
    print(f"Best Score: {grid_search.best_score_:.6f}")
    print(f"Best Params: {grid_search.best_params_}")
    print("=" * 60)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }


def random_search_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    param_distributions: Dict[str, List],
    n_iter: int = 100,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Random Search with Cross Validation
    
    Parameters:
    -----------
    model : estimator
        Model to tune
    X : array-like
        Features
    y : array-like
        Target
    param_distributions : dict
        Parameter distributions
    n_iter : int, default=100
        Number of iterations
    cv : int, default=5
        Number of folds
    scoring : str, optional
        Scoring metric
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random state
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    dict
        Results with best params and score
    """
    print("=" * 60)
    print("🎲 Random Search CV")
    print("=" * 60)
    print(f"Parameter distributions: {param_distributions}")
    print(f"Iterations: {n_iter}")
    print(f"CV folds: {cv}")
    print(f"Scoring: {scoring or 'default'}")
    print()
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )
    
    random_search.fit(X, y)
    
    print("=" * 60)
    print("✅ Random Search Complete!")
    print("=" * 60)
    print(f"Best Score: {random_search.best_score_:.6f}")
    print(f"Best Params: {random_search.best_params_}")
    print("=" * 60)
    
    return {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'best_estimator': random_search.best_estimator_,
        'cv_results': pd.DataFrame(random_search.cv_results_)
    }


def bayesian_optimization(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    param_space_func: Callable,
    n_trials: int = 100,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    task: str = 'regression',
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Bayesian Optimization ด้วย Optuna
    
    Parameters:
    -----------
    model_class : class
        Model class (not instance)
    X : array-like
        Features
    y : array-like
        Target
    param_space_func : callable
        Function that takes trial and returns param dict
    n_trials : int, default=100
        Number of trials
    cv : int, default=5
        Number of folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric
    task : str, default='regression'
        'regression' or 'classification'
    random_state : int, default=42
        Random state
    verbose : bool, default=True
        Print progress
    
    Returns:
    --------
    dict
        Results with best params and score
    
    Examples:
    ---------
    >>> from lightgbm import LGBMRegressor
    >>> 
    >>> def param_space(trial):
    ...     return {
    ...         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    ...         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    ...         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    ...     }
    >>> 
    >>> result = bayesian_optimization(
    ...     LGBMRegressor,
    ...     X_train, y_train,
    ...     param_space,
    ...     n_trials=50
    ... )
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for Bayesian optimization. "
                         "Install with: pip install optuna")
    
    if verbose:
        print("=" * 60)
        print("🎯 Bayesian Optimization (Optuna)")
        print("=" * 60)
        print(f"Trials: {n_trials}")
        print(f"CV folds: {cv}")
        print(f"Scoring: {scoring}")
        print()
    
    def objective(trial):
        params = param_space_func(trial)
        model = model_class(**params, random_state=random_state)
        
        # Cross validation
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        return scores.mean()
    
    # Create study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction='maximize' if 'neg' in scoring else 'minimize',
        sampler=sampler
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=verbose
    )
    
    if verbose:
        print("=" * 60)
        print("✅ Optimization Complete!")
        print("=" * 60)
        print(f"Best Score: {study.best_value:.6f}")
        print(f"Best Params: {study.best_params}")
        print("=" * 60)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study,
        'trials_df': study.trials_dataframe()
    }


def tune_hyperparameters(
    model,
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'random',
    param_space: Optional[Dict] = None,
    model_type: Optional[str] = None,
    n_iter: int = 100,
    cv: int = 5,
    scoring: Optional[str] = None,
    task: str = 'regression',
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    หา hyperparameters ที่ดีที่สุด (wrapper function)
    
    Parameters:
    -----------
    model : estimator or class
        Model to tune
    X : array-like
        Features
    y : array-like
        Target
    method : str, default='random'
        'grid', 'random', 'bayesian'
    param_space : dict, optional
        Parameter space (ถ้าไม่ใส่จะใช้ default)
    model_type : str, optional
        'lightgbm', 'xgboost', 'catboost', 'random_forest'
    n_iter : int, default=100
        Number of iterations (for random/bayesian)
    cv : int, default=5
        Number of folds
    scoring : str, optional
        Scoring metric
    task : str, default='regression'
        'regression' or 'classification'
    random_state : int, default=42
        Random state
    verbose : bool, default=True
        Print progress
    
    Returns:
    --------
    dict
        Results with best params and score
    """
    # ถ้าไม่ระบุ param_space ให้ใช้ default
    if param_space is None:
        if model_type == 'lightgbm':
            param_space = suggest_params_lightgbm('default', task)
        elif model_type == 'xgboost':
            param_space = suggest_params_xgboost('default', task)
        elif model_type == 'catboost':
            param_space = suggest_params_catboost('default', task)
        elif model_type == 'random_forest':
            param_space = suggest_params_random_forest('default')
        else:
            raise ValueError("Please provide param_space or model_type")
    
    # Default scoring
    if scoring is None:
        scoring = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'
    
    # Tune
    if method == 'grid':
        return grid_search_cv(
            model, X, y, param_space, cv, scoring,
            verbose=1 if verbose else 0
        )
    
    elif method == 'random':
        return random_search_cv(
            model, X, y, param_space, n_iter, cv, scoring,
            random_state=random_state,
            verbose=1 if verbose else 0
        )
    
    elif method == 'bayesian':
        if not OPTUNA_AVAILABLE:
            print("⚠️  Optuna not available, falling back to Random Search")
            return random_search_cv(
                model, X, y, param_space, n_iter, cv, scoring,
                random_state=random_state,
                verbose=1 if verbose else 0
            )
        
        # For Bayesian, need to define param space function
        # This is a simplified version - for full control use bayesian_optimization directly
        raise NotImplementedError(
            "For Bayesian optimization, please use bayesian_optimization() directly"
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Example 1: Random Search
    print("\n" + "="*60)
    print("📊 RANDOM SEARCH EXAMPLE")
    print("="*60)
    
    param_space = suggest_params_random_forest('narrow')
    model = RandomForestRegressor(random_state=42)
    
    result = tune_hyperparameters(
        model, X_train, y_train,
        method='random',
        param_space=param_space,
        n_iter=10,  # Small for demo
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=True
    )
    
    print(f"\nBest params: {result['best_params']}")
    print(f"Best score: {result['best_score']:.6f}")