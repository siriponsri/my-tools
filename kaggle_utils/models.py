"""
Model Wrappers and Training Utilities
รวม wrappers สำหรับ sklearn, LightGBM, XGBoost, CatBoost พร้อม cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== SCIKIT-LEARN WRAPPERS ====================

class SKLearnWrapper:
    """
    Universal Scikit-Learn Model Wrapper with Cross-Validation
    ใช้ได้กับ sklearn model ใดก็ได้
    
    Parameters:
    -----------
    model : sklearn estimator
        โมเดลที่ต้องการ train
    n_splits : int
        จำนวน folds สำหรับ cross-validation
    random_state : int
        Random seed
    task : str
        'regression' หรือ 'classification'
    verbose : bool
        แสดง progress bar หรือไม่
    
    Attributes:
    -----------
    models : list
        โมเดลแต่ละ fold
    scores : list
        Score แต่ละ fold
    oof_predictions : np.array
        Out-of-fold predictions
    test_predictions : np.array
        Predictions สำหรับ test set
    """
    
    def __init__(self, model, n_splits=5, random_state=42, task='regression', verbose=True):
        self.model = model
        self.task = task
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.models = []
        self.scores = []
        self.oof_predictions = None
        self.test_predictions = None
    
    def train(self, X, y, X_test=None):
        """
        Train model with K-Fold Cross-Validation
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Training features
        y : pd.Series or np.array
            Training target
        X_test : pd.DataFrame or np.array, optional
            Test features
        
        Returns:
        --------
        self
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_predictions = np.zeros(len(X))
        test_predictions = []
        
        print(f"{'='*60}")
        print(f"🚀 Training {self.model.__class__.__name__} with {self.n_splits}-Fold CV")
        print(f"{'='*60}")
        
        # Progress bar for folds
        fold_iterator = tqdm(
            enumerate(kfold.split(X), 1),
            total=self.n_splits,
            desc="Training Folds",
            disable=not self.verbose,
            bar_format='{l_bar}{bar:30}{r_bar}'
        )
        
        for fold, (train_idx, val_idx) in fold_iterator:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone model for each fold
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            # Predictions
            if self.task == 'classification' and hasattr(model, 'predict_proba'):
                self.oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
                if X_test is not None:
                    test_predictions.append(model.predict_proba(X_test)[:, 1])
            else:
                self.oof_predictions[val_idx] = model.predict(X_val)
                if X_test is not None:
                    test_predictions.append(model.predict(X_test))
            
            # Calculate score
            if self.task == 'regression':
                score = np.sqrt(mean_squared_error(y_val, self.oof_predictions[val_idx]))
                fold_iterator.set_postfix({'Fold': fold, 'RMSE': f'{score:.4f}'})
            else:
                score = roc_auc_score(y_val, self.oof_predictions[val_idx])
                fold_iterator.set_postfix({'Fold': fold, 'AUC': f'{score:.4f}'})
            
            self.scores.append(score)
            self.models.append(model)
        
        # Overall score
        if self.task == 'regression':
            overall_score = np.sqrt(mean_squared_error(y, self.oof_predictions))
            print(f"\n📊 Overall RMSE: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        else:
            overall_score = roc_auc_score(y, self.oof_predictions)
            print(f"\n📊 Overall AUC: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        
        # Average test predictions
        if test_predictions:
            self.test_predictions = np.mean(test_predictions, axis=0)
        
        return self
    
    def predict(self, X):
        """Predict using average of all fold models"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        predictions = []
        for model in self.models:
            if self.task == 'classification' and hasattr(model, 'predict_proba'):
                predictions.append(model.predict_proba(X)[:, 1])
            else:
                predictions.append(model.predict(X))
        
        return np.mean(predictions, axis=0)
    
    def get_feature_importance(self, feature_names=None):
        """Get average feature importance across folds"""
        if not hasattr(self.models[0], 'feature_importances_'):
            print("⚠️  Model doesn't have feature_importances_ attribute")
            return None
        
        importances = np.mean([model.feature_importances_ for model in self.models], axis=0)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


class RandomForestWrapper:
    """Random Forest Wrapper with optimized defaults"""
    
    def __init__(self, n_splits=5, random_state=42, task='regression', verbose=True, **kwargs):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': random_state
        }
        params = {**default_params, **kwargs}
        
        if task == 'regression':
            model = RandomForestRegressor(**params)
        else:
            model = RandomForestClassifier(**params)
        
        self.wrapper = SKLearnWrapper(model, n_splits, random_state, task, verbose)
    
    def train(self, X, y, X_test=None):
        return self.wrapper.train(X, y, X_test)
    
    def predict(self, X):
        return self.wrapper.predict(X)
    
    @property
    def oof_predictions(self):
        return self.wrapper.oof_predictions
    
    @property
    def test_predictions(self):
        return self.wrapper.test_predictions
    
    @property
    def models(self):
        return self.wrapper.models
    
    @property
    def scores(self):
        return self.wrapper.scores
    
    def get_feature_importance(self, feature_names=None):
        return self.wrapper.get_feature_importance(feature_names)


class RidgeWrapper:
    """Ridge Regression Wrapper"""
    
    def __init__(self, alpha=1.0, n_splits=5, random_state=42, verbose=True):
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha, random_state=random_state)
        self.wrapper = SKLearnWrapper(model, n_splits, random_state, 'regression', verbose)
    
    def train(self, X, y, X_test=None):
        return self.wrapper.train(X, y, X_test)
    
    def predict(self, X):
        return self.wrapper.predict(X)
    
    @property
    def oof_predictions(self):
        return self.wrapper.oof_predictions
    
    @property
    def test_predictions(self):
        return self.wrapper.test_predictions
    
    @property
    def models(self):
        return self.wrapper.models
    
    @property
    def scores(self):
        return self.wrapper.scores


class LassoWrapper:
    """Lasso Regression Wrapper"""
    
    def __init__(self, alpha=1.0, n_splits=5, random_state=42, verbose=True):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
        self.wrapper = SKLearnWrapper(model, n_splits, random_state, 'regression', verbose)
    
    def train(self, X, y, X_test=None):
        return self.wrapper.train(X, y, X_test)
    
    def predict(self, X):
        return self.wrapper.predict(X)
    
    @property
    def oof_predictions(self):
        return self.wrapper.oof_predictions
    
    @property
    def test_predictions(self):
        return self.wrapper.test_predictions
    
    @property
    def models(self):
        return self.wrapper.models
    
    @property
    def scores(self):
        return self.wrapper.scores


class ElasticNetWrapper:
    """ElasticNet Wrapper"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, n_splits=5, random_state=42, verbose=True):
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)
        self.wrapper = SKLearnWrapper(model, n_splits, random_state, 'regression', verbose)
    
    def train(self, X, y, X_test=None):
        return self.wrapper.train(X, y, X_test)
    
    def predict(self, X):
        return self.wrapper.predict(X)
    
    @property
    def oof_predictions(self):
        return self.wrapper.oof_predictions
    
    @property
    def test_predictions(self):
        return self.wrapper.test_predictions
    
    @property
    def models(self):
        return self.wrapper.models
    
    @property
    def scores(self):
        return self.wrapper.scores


# ==================== LIGHTGBM WRAPPER ====================

class LGBWrapper:
    """
    LightGBM Wrapper with Cross-Validation
    
    Parameters:
    -----------
    params : dict
        LightGBM parameters
    n_splits : int
        จำนวน folds
    random_state : int
        Random seed
    task : str
        'regression' หรือ 'classification'
    verbose : bool
        แสดง progress bar หรือไม่
    """
    
    def __init__(self, params=None, n_splits=5, random_state=42, task='regression', verbose=True):
        self.task = task
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.models = []
        self.scores = []
        
        default_params = {
            'objective': 'regression' if task == 'regression' else 'binary',
            'metric': 'rmse' if task == 'regression' else 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        self.params = {**default_params, **(params or {})}
    
    def train(self, X, y, X_test=None, categorical_features=None):
        """Train with K-Fold CV"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
            return None
        
        # Convert to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_predictions = np.zeros(len(X))
        test_predictions = []
        
        print(f"{'='*60}")
        print(f"🚀 Training LightGBM with {self.n_splits}-Fold CV")
        print(f"{'='*60}")
        
        # Progress bar for folds
        fold_iterator = tqdm(
            enumerate(kfold.split(X), 1),
            total=self.n_splits,
            desc="🌲 LightGBM Training",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for fold, (train_idx, val_idx) in fold_iterator:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.task == 'regression':
                model = lgb.LGBMRegressor(**self.params)
            else:
                model = lgb.LGBMClassifier(**self.params)
            
            # Suppress LightGBM's own progress
            callbacks = [lgb.early_stopping(50)]
            if not self.verbose:
                callbacks.append(lgb.log_evaluation(0))
            else:
                callbacks.append(lgb.log_evaluation(0))  # Suppress individual iterations
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            self.oof_predictions[val_idx] = model.predict(X_val)
            
            if X_test is not None:
                test_predictions.append(model.predict(X_test))
            
            if self.task == 'regression':
                score = np.sqrt(mean_squared_error(y_val, self.oof_predictions[val_idx]))
                fold_iterator.set_postfix({'Fold': fold, 'RMSE': f'{score:.4f}'})
            else:
                score = roc_auc_score(y_val, self.oof_predictions[val_idx])
                fold_iterator.set_postfix({'Fold': fold, 'AUC': f'{score:.4f}'})
            
            self.scores.append(score)
            self.models.append(model)
        
        # Overall score
        if self.task == 'regression':
            overall_score = np.sqrt(mean_squared_error(y, self.oof_predictions))
            print(f"\n📊 Overall RMSE: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        else:
            overall_score = roc_auc_score(y, self.oof_predictions)
            print(f"\n📊 Overall AUC: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        
        self.test_predictions = np.mean(test_predictions, axis=0) if test_predictions else None
        
        return self
    
    def predict(self, X):
        """Predict using average of all fold models"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
    
    def get_feature_importance(self, feature_names, top_n=20):
        """Get average feature importance"""
        import matplotlib.pyplot as plt
        
        importances = np.mean([model.feature_importances_ for model in self.models], axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (LightGBM)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df


# ==================== XGBOOST WRAPPER ====================

class XGBWrapper:
    """XGBoost Wrapper with Cross-Validation"""
    
    def __init__(self, params=None, n_splits=5, random_state=42, task='regression', verbose=True):
        self.task = task
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.models = []
        self.scores = []
        
        default_params = {
            'objective': 'reg:squarederror' if task == 'regression' else 'binary:logistic',
            'eval_metric': 'rmse' if task == 'regression' else 'auc',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        self.params = {**default_params, **(params or {})}
    
    def train(self, X, y, X_test=None):
        """Train with K-Fold CV"""
        try:
            import xgboost as xgb
        except ImportError:
            print("⚠️  XGBoost not installed. Install with: pip install xgboost")
            return None
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_predictions = np.zeros(len(X))
        test_predictions = []
        
        print(f"{'='*60}")
        print(f"🚀 Training XGBoost with {self.n_splits}-Fold CV")
        print(f"{'='*60}")
        
        # Progress bar for folds
        fold_iterator = tqdm(
            enumerate(kfold.split(X), 1),
            total=self.n_splits,
            desc="🎯 XGBoost Training",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for fold, (train_idx, val_idx) in fold_iterator:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.task == 'regression':
                model = xgb.XGBRegressor(**self.params)
            else:
                model = xgb.XGBClassifier(**self.params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False  # Suppress XGBoost's own output
            )
            
            self.oof_predictions[val_idx] = model.predict(X_val)
            
            if X_test is not None:
                test_predictions.append(model.predict(X_test))
            
            if self.task == 'regression':
                score = np.sqrt(mean_squared_error(y_val, self.oof_predictions[val_idx]))
                fold_iterator.set_postfix({'Fold': fold, 'RMSE': f'{score:.4f}'})
            else:
                score = roc_auc_score(y_val, self.oof_predictions[val_idx])
                fold_iterator.set_postfix({'Fold': fold, 'AUC': f'{score:.4f}'})
            
            self.scores.append(score)
            self.models.append(model)
        
        if self.task == 'regression':
            overall_score = np.sqrt(mean_squared_error(y, self.oof_predictions))
            print(f"\n📊 Overall RMSE: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        else:
            overall_score = roc_auc_score(y, self.oof_predictions)
            print(f"\n📊 Overall AUC: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        
        self.test_predictions = np.mean(test_predictions, axis=0) if test_predictions else None
        
        return self
    
    def predict(self, X):
        """Predict using average of all fold models"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
    
    def get_feature_importance(self, feature_names, top_n=20):
        """Get average feature importance"""
        import matplotlib.pyplot as plt
        
        importances = np.mean([model.feature_importances_ for model in self.models], axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df


# ==================== CATBOOST WRAPPER ====================

class CatBoostWrapper:
    """CatBoost Wrapper with Cross-Validation"""
    
    def __init__(self, params=None, n_splits=5, random_state=42, task='regression', verbose=True):
        self.task = task
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.models = []
        self.scores = []
        
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 7,
            'l2_leaf_reg': 3,
            'random_seed': random_state,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        self.params = {**default_params, **(params or {})}
    
    def train(self, X, y, X_test=None, cat_features=None):
        """Train with K-Fold CV"""
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            print("⚠️  CatBoost not installed. Install with: pip install catboost")
            return None
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_predictions = np.zeros(len(X))
        test_predictions = []
        
        print(f"{'='*60}")
        print(f"🚀 Training CatBoost with {self.n_splits}-Fold CV")
        print(f"{'='*60}")
        
        # Progress bar for folds
        fold_iterator = tqdm(
            enumerate(kfold.split(X), 1),
            total=self.n_splits,
            desc="🐱 CatBoost Training",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for fold, (train_idx, val_idx) in fold_iterator:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.task == 'regression':
                model = CatBoostRegressor(**self.params)
            else:
                model = CatBoostClassifier(**self.params)
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
                verbose=False
            )
            
            self.oof_predictions[val_idx] = model.predict(X_val)
            
            if X_test is not None:
                test_predictions.append(model.predict(X_test))
            
            if self.task == 'regression':
                score = np.sqrt(mean_squared_error(y_val, self.oof_predictions[val_idx]))
                fold_iterator.set_postfix({'Fold': fold, 'RMSE': f'{score:.4f}'})
            else:
                score = roc_auc_score(y_val, self.oof_predictions[val_idx])
                fold_iterator.set_postfix({'Fold': fold, 'AUC': f'{score:.4f}'})
            
            self.scores.append(score)
            self.models.append(model)
        
        if self.task == 'regression':
            overall_score = np.sqrt(mean_squared_error(y, self.oof_predictions))
            print(f"\n📊 Overall RMSE: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        else:
            overall_score = roc_auc_score(y, self.oof_predictions)
            print(f"\n📊 Overall AUC: {overall_score:.4f} (±{np.std(self.scores):.4f})")
        
        self.test_predictions = np.mean(test_predictions, axis=0) if test_predictions else None
        
        return self


# ==================== UTILITY FUNCTIONS ====================

def quick_model_comparison(X, y, cv=5, random_state=42, scoring='neg_mean_squared_error', verbose=True):
    """
    เปรียบเทียบ regression models หลายๆ ตัวอย่างรวดเร็ว
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
    scoring : str
        Scoring metric
    verbose : bool
        Show progress bar
    
    Returns:
    --------
    pd.DataFrame
        Comparison results sorted by performance
    """
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor, 
        ExtraTreesRegressor, AdaBoostRegressor
    )
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=random_state),
        'Lasso': Lasso(random_state=random_state, max_iter=10000),
        'ElasticNet': ElasticNet(random_state=random_state, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=random_state),
        'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }
    
    results = []
    print("🔄 Comparing regression models...")
    print(f"{'='*60}")
    
    # Progress bar for models
    model_iterator = tqdm(
        models.items(),
        desc="Testing Models",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for name, model in model_iterator:
        try:
            model_iterator.set_postfix({'Current': name[:20]})
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            rmse = np.sqrt(-scores.mean())
            std = np.sqrt(scores.std())
            results.append({
                'Model': name, 
                'RMSE': rmse, 
                'Std': std,
                'Score': -scores.mean()
            })
        except Exception as e:
            if verbose:
                print(f"\n⚠️  {name}: Error - {str(e)[:50]}")
    
    results_df = pd.DataFrame(results).sort_values('RMSE')
    
    print(f"\n{'='*60}")
    print(f"✅ Best model: {results_df.iloc[0]['Model']}")
    print(f"   RMSE: {results_df.iloc[0]['RMSE']:.4f}")
    
    return results_df


def quick_classification_comparison(X, y, cv=5, random_state=42, scoring='roc_auc', verbose=True):
    """เปรียบเทียบ classification models หลายๆ ตัว"""
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        ExtraTreesClassifier, AdaBoostClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    
    results = []
    print("🔄 Comparing classification models...")
    print(f"{'='*60}")
    
    # Progress bar for models
    model_iterator = tqdm(
        models.items(),
        desc="Testing Models",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for name, model in model_iterator:
        try:
            model_iterator.set_postfix({'Current': name[:20]})
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            auc = scores.mean()
            std = scores.std()
            results.append({'Model': name, 'AUC': auc, 'Std': std})
        except Exception as e:
            if verbose:
                print(f"\n⚠️  {name}: Error - {str(e)[:50]}")
    
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Best model: {results_df.iloc[0]['Model']}")
    print(f"   AUC: {results_df.iloc[0]['AUC']:.4f}")
    
    return results_df


def compare_scalers(X, y, model, cv=5, scoring='neg_mean_squared_error', verbose=True):
    """เปรียบเทียบ scalers ต่างๆ"""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'No Scaling': None
    }
    
    results = {}
    print("🔄 Comparing scalers...")
    print(f"{'='*60}")
    
    # Progress bar
    scaler_iterator = tqdm(
        scalers.items(),
        desc="Testing Scalers",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for name, scaler in scaler_iterator:
        scaler_iterator.set_postfix({'Current': name})
        
        if scaler:
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns if hasattr(X, 'columns') else None)
        else:
            X_scaled = X
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
        rmse = np.sqrt(-scores.mean())
        results[name] = rmse
    
    best_scaler = min(results, key=results.get)
    
    print(f"\n{'='*60}")
    print(f"✅ Best scaler: {best_scaler}")
    print(f"   RMSE: {results[best_scaler]:.4f}")
    
    # Print all results
    for name, rmse in sorted(results.items(), key=lambda x: x[1]):
        print(f"   {name:20s}: {rmse:.4f}")
    
    return best_scaler, results


def create_pipeline(estimator, scale=True, scaler='standard'):
    """
    สร้าง sklearn pipeline
    
    Parameters:
    -----------
    estimator : sklearn estimator
        โมเดลที่ต้องการใช้
    scale : bool
        ใส่ scaler หรือไม่
    scaler : str
        'standard', 'minmax', 'robust'
    
    Returns:
    --------
    Pipeline
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    steps = []
    
    if scale:
        if scaler == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaler == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        elif scaler == 'robust':
            steps.append(('scaler', RobustScaler()))
    
    steps.append(('estimator', estimator))
    
    pipeline = Pipeline(steps)
    print(f"✅ Created pipeline with {len(steps)} steps")
    
    return pipeline


print("✅ Models module loaded successfully!")
print("💡 All wrappers now include progress bars with tqdm!")