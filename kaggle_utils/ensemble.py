"""
Ensemble Methods
รวมวิธีการ ensemble models เพื่อเพิ่มความแม่นยำ
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== WEIGHTED ENSEMBLE ====================

class WeightedEnsemble(BaseEstimator):
    """
    Weighted Average Ensemble
    รวม predictions จากหลาย models ด้วยน้ำหนัก
    
    Parameters:
    -----------
    models : list
        List of trained models
    weights : list, optional
        น้ำหนักของแต่ละ model (ถ้าไม่ระบุจะเท่ากันหมด)
    task : str
        'regression' หรือ 'classification'
    verbose : bool
        แสดง progress หรือไม่
    
    Example:
    --------
    >>> ensemble = WeightedEnsemble(
    ...     models=[model1, model2, model3],
    ...     weights=[0.5, 0.3, 0.2]
    ... )
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(self, models, weights=None, task='regression', verbose=True):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        self.task = task
        self.verbose = verbose
        
        # Validate weights
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        if not np.isclose(sum(self.weights), 1.0):
            print(f"⚠️  Warning: Weights sum to {sum(self.weights):.4f}, normalizing to 1.0")
            self.weights = [w / sum(self.weights) for w in self.weights]
    
    def fit(self, X, y):
        """Train all models"""
        print(f"{'='*60}")
        print(f"🎯 Training Weighted Ensemble ({len(self.models)} models)")
        print(f"{'='*60}")
        print(f"Weights: {[f'{w:.3f}' for w in self.weights]}")
        
        model_iterator = tqdm(
            enumerate(self.models, 1),
            total=len(self.models),
            desc="Training Models",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for idx, model in model_iterator:
            model_iterator.set_postfix({'Model': idx})
            model.fit(X, y)
        
        print(f"✅ All models trained successfully!")
        return self
    
    def predict(self, X):
        """Predict using weighted average"""
        predictions = []
        
        model_iterator = tqdm(
            self.models,
            desc="🔮 Predicting",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for model in model_iterator:
            if self.task == 'classification' and hasattr(model, 'predict_proba'):
                predictions.append(model.predict_proba(X)[:, 1])
            else:
                predictions.append(model.predict(X))
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred
    
    def predict_proba(self, X):
        """For classification - return probabilities"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.predict(X)


# ==================== STACKING ENSEMBLE ====================

class StackingEnsemble(BaseEstimator):
    """
    Stacking Ensemble with Meta-Learner
    ใช้ predictions จาก base models เป็น features สำหรับ meta model
    
    Parameters:
    -----------
    base_models : list
        List of base models (level 0)
    meta_model : sklearn estimator
        Meta model (level 1)
    cv : int
        Number of folds สำหรับสร้าง meta features
    task : str
        'regression' หรือ 'classification'
    verbose : bool
        แสดง progress หรือไม่
    
    Example:
    --------
    >>> from sklearn.linear_model import Ridge
    >>> stacking = StackingEnsemble(
    ...     base_models=[lgb_model, xgb_model, rf_model],
    ...     meta_model=Ridge(),
    ...     cv=5
    ... )
    >>> stacking.fit(X_train, y_train)
    >>> predictions = stacking.predict(X_test)
    """
    
    def __init__(self, base_models, meta_model, cv=5, task='regression', 
                 random_state=42, verbose=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.task = task
        self.random_state = random_state
        self.verbose = verbose
        self.meta_features_train = None
    
    def fit(self, X, y):
        """
        Train base models และ meta model
        
        Steps:
        1. Train base models ด้วย CV และสร้าง out-of-fold predictions
        2. ใช้ OOF predictions เป็น meta features
        3. Train meta model บน meta features
        4. Retrain base models บน full data
        """
        print(f"{'='*60}")
        print(f"🏗️  Training Stacking Ensemble")
        print(f"{'='*60}")
        print(f"Base models: {len(self.base_models)}")
        print(f"Meta model: {self.meta_model.__class__.__name__}")
        print(f"CV folds: {self.cv}")
        
        # Convert to DataFrame/Series
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Step 1: Create meta features using CV
        print(f"\n📊 Step 1: Creating meta features with {self.cv}-fold CV...")
        
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        base_model_iterator = tqdm(
            enumerate(self.base_models, 1),
            total=len(self.base_models),
            desc="Base Models",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for model_idx, base_model in base_model_iterator:
            base_model_iterator.set_postfix({'Model': model_idx})
            
            # CV loop for this base model
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                # Clone and train
                model = clone(base_model)
                model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                if self.task == 'classification' and hasattr(model, 'predict_proba'):
                    meta_features[val_idx, model_idx - 1] = model.predict_proba(X_val_fold)[:, 1]
                else:
                    meta_features[val_idx, model_idx - 1] = model.predict(X_val_fold)
        
        self.meta_features_train = meta_features
        
        # Step 2: Train meta model
        print(f"\n🎯 Step 2: Training meta model...")
        self.meta_model.fit(meta_features, y)
        
        # Calculate OOF score
        meta_pred = self.meta_model.predict(meta_features)
        if self.task == 'regression':
            oof_score = np.sqrt(mean_squared_error(y, meta_pred))
            print(f"✅ Stacking OOF RMSE: {oof_score:.4f}")
        else:
            oof_score = roc_auc_score(y, meta_pred)
            print(f"✅ Stacking OOF AUC: {oof_score:.4f}")
        
        # Step 3: Retrain base models on full data
        print(f"\n🔄 Step 3: Retraining base models on full data...")
        
        retrain_iterator = tqdm(
            self.base_models,
            desc="Retraining",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for base_model in retrain_iterator:
            base_model.fit(X, y)
        
        print(f"\n{'='*60}")
        print(f"✅ Stacking Ensemble training complete!")
        print(f"{'='*60}")
        
        return self
    
    def predict(self, X):
        """Predict using stacked model"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Get base model predictions
        meta_features_test = np.zeros((X.shape[0], len(self.base_models)))
        
        model_iterator = tqdm(
            enumerate(self.base_models),
            total=len(self.base_models),
            desc="🔮 Base Predictions",
            disable=not self.verbose,
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
        )
        
        for idx, base_model in model_iterator:
            if self.task == 'classification' and hasattr(base_model, 'predict_proba'):
                meta_features_test[:, idx] = base_model.predict_proba(X)[:, 1]
            else:
                meta_features_test[:, idx] = base_model.predict(X)
        
        # Meta model prediction
        return self.meta_model.predict(meta_features_test)


# ==================== VOTING ENSEMBLE ====================

def create_voting_ensemble(models, model_names=None, voting='soft', weights=None, task='classification'):
    """
    สร้าง Voting Ensemble (sklearn wrapper)
    
    Parameters:
    -----------
    models : list
        List of models
    model_names : list, optional
        ชื่อของแต่ละ model
    voting : str
        'soft' (average probabilities) หรือ 'hard' (majority vote)
    weights : list, optional
        น้ำหนักของแต่ละ model
    task : str
        'classification' หรือ 'regression'
    
    Returns:
    --------
    VotingClassifier or VotingRegressor
    
    Example:
    --------
    >>> voting = create_voting_ensemble(
    ...     models=[rf_model, lgb_model, xgb_model],
    ...     model_names=['rf', 'lgb', 'xgb'],
    ...     voting='soft',
    ...     weights=[0.3, 0.4, 0.3]
    ... )
    >>> voting.fit(X_train, y_train)
    >>> predictions = voting.predict(X_test)
    """
    if model_names is None:
        model_names = [f'model_{i}' for i in range(len(models))]
    
    estimators = list(zip(model_names, models))
    
    if task == 'classification':
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
    else:
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1
        )
    
    print(f"✅ Created {task} voting ensemble with {len(models)} models")
    if weights:
        print(f"   Weights: {weights}")
    
    return ensemble


# ==================== BLENDING ====================

def blend_predictions(predictions_list, weights=None, method='average'):
    """
    Blend predictions จากหลาย models
    
    Parameters:
    -----------
    predictions_list : list of arrays
        List ของ predictions จากแต่ละ model
    weights : list, optional
        น้ำหนักของแต่ละ prediction (ถ้าไม่ระบุจะเท่ากันหมด)
    method : str
        'average' (weighted average)
        'rank' (rank averaging - ดีสำหรับ predictions ที่ scale ต่างกัน)
        'geometric' (geometric mean)
        'harmonic' (harmonic mean)
        'median' (median)
    
    Returns:
    --------
    np.array
        Blended predictions
    
    Example:
    --------
    >>> pred1 = model1.predict(X_test)
    >>> pred2 = model2.predict(X_test)
    >>> pred3 = model3.predict(X_test)
    >>> 
    >>> # Average
    >>> final = blend_predictions([pred1, pred2, pred3], weights=[0.5, 0.3, 0.2])
    >>> 
    >>> # Rank averaging (ดีเมื่อ predictions มี scale ต่างกัน)
    >>> final = blend_predictions([pred1, pred2, pred3], method='rank')
    """
    predictions_array = np.array(predictions_list)
    
    if weights is None:
        weights = [1/len(predictions_list)] * len(predictions_list)
    
    # Validate
    if len(weights) != len(predictions_list):
        raise ValueError("Number of weights must match number of predictions")
    
    if not np.isclose(sum(weights), 1.0):
        print(f"⚠️  Warning: Weights sum to {sum(weights):.4f}, normalizing to 1.0")
        weights = [w / sum(weights) for w in weights]
    
    print(f"🎯 Blending {len(predictions_list)} predictions using '{method}' method")
    print(f"   Weights: {[f'{w:.3f}' for w in weights]}")
    
    if method == 'average':
        # Weighted average
        blended = np.average(predictions_array, axis=0, weights=weights)
        
    elif method == 'rank':
        # Rank averaging (แปลงเป็น rank ก่อน average)
        ranks = np.zeros_like(predictions_array)
        for i in range(len(predictions_list)):
            ranks[i] = pd.Series(predictions_array[i]).rank().values
        blended = np.average(ranks, axis=0, weights=weights)
        
    elif method == 'geometric':
        # Geometric mean (ดีสำหรับ predictions ที่เป็น probabilities)
        weighted_preds = [pred ** w for pred, w in zip(predictions_array, weights)]
        blended = np.prod(weighted_preds, axis=0) ** (1/sum(weights))
        
    elif method == 'harmonic':
        # Harmonic mean
        weighted_reciprocals = [w / pred for pred, w in zip(predictions_array, weights)]
        blended = sum(weights) / np.sum(weighted_reciprocals, axis=0)
        
    elif method == 'median':
        # Median (ignore weights)
        blended = np.median(predictions_array, axis=0)
        print(f"   Note: Weights ignored for median")
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'average', 'rank', 'geometric', 'harmonic', 'median'")
    
    print(f"✅ Blending complete!")
    return blended


# ==================== ADVANCED BLENDING ====================

def optimize_blend_weights(predictions_list, y_true, method='nelder-mead', metric='rmse'):
    """
    หาน้ำหนักที่ดีที่สุดสำหรับ blending โดยใช้ optimization
    
    Parameters:
    -----------
    predictions_list : list of arrays
        List ของ predictions จากแต่ละ model
    y_true : array
        Target values จริง
    method : str
        Optimization method: 'nelder-mead', 'powell', 'bfgs'
    metric : str
        'rmse' (regression) หรือ 'auc' (classification)
    
    Returns:
    --------
    np.array
        Optimal weights
    
    Example:
    --------
    >>> # หา weights ที่ดีที่สุดจาก validation set
    >>> weights = optimize_blend_weights(
    ...     [pred1_val, pred2_val, pred3_val],
    ...     y_val,
    ...     metric='rmse'
    ... )
    >>> # ใช้ weights นี้กับ test set
    >>> final_pred = blend_predictions(
    ...     [pred1_test, pred2_test, pred3_test],
    ...     weights=weights
    ... )
    """
    from scipy.optimize import minimize
    
    print(f"🔍 Optimizing blend weights using '{method}' method...")
    
    def objective(weights):
        """Objective function to minimize"""
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Blend predictions
        blended = np.average(predictions_list, axis=0, weights=weights)
        
        # Calculate metric
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, blended))
        elif metric == 'auc':
            return -roc_auc_score(y_true, blended)  # Negative because we minimize
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Initial weights (equal)
    n_models = len(predictions_list)
    initial_weights = np.ones(n_models) / n_models
    
    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    
    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method=method,
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x / result.x.sum()  # Normalize
    optimal_score = result.fun if metric == 'rmse' else -result.fun
    
    print(f"✅ Optimization complete!")
    print(f"   Optimal weights: {[f'{w:.3f}' for w in optimal_weights]}")
    print(f"   Optimal {metric.upper()}: {optimal_score:.4f}")
    
    return optimal_weights


def create_meta_features(models, X_train, y_train, X_test, cv=5, task='regression', 
                        random_state=42, verbose=True):
    """
    สร้าง meta features สำหรับ stacking (without training meta model)
    ใช้เมื่อต้องการควบคุม stacking ด้วยตัวเอง
    
    Parameters:
    -----------
    models : list
        List of base models
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    cv : int
        Number of folds
    task : str
        'regression' หรือ 'classification'
    random_state : int
        Random seed
    verbose : bool
        แสดง progress
    
    Returns:
    --------
    tuple
        (meta_features_train, meta_features_test)
    
    Example:
    --------
    >>> # สร้าง meta features
    >>> meta_train, meta_test = create_meta_features(
    ...     models=[lgb_model, xgb_model, rf_model],
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, cv=5
    ... )
    >>> 
    >>> # Train meta model ด้วยตัวเอง
    >>> meta_model = Ridge()
    >>> meta_model.fit(meta_train, y_train)
    >>> final_pred = meta_model.predict(meta_test)
    """
    print(f"{'='*60}")
    print(f"🏗️  Creating meta features")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"CV folds: {cv}")
    
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    meta_train = np.zeros((X_train.shape[0], len(models)))
    meta_test = np.zeros((X_test.shape[0], len(models)))
    
    model_iterator = tqdm(
        enumerate(models, 1),
        total=len(models),
        desc="Creating Meta Features",
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    for model_idx, model in model_iterator:
        model_iterator.set_postfix({'Model': model_idx})
        
        # Out-of-fold predictions for train
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)
            
            if task == 'classification' and hasattr(model_clone, 'predict_proba'):
                meta_train[val_idx, model_idx - 1] = model_clone.predict_proba(X_val)[:, 1]
            else:
                meta_train[val_idx, model_idx - 1] = model_clone.predict(X_val)
        
        # Test predictions (train on full data)
        model.fit(X_train, y_train)
        if task == 'classification' and hasattr(model, 'predict_proba'):
            meta_test[:, model_idx - 1] = model.predict_proba(X_test)[:, 1]
        else:
            meta_test[:, model_idx - 1] = model.predict(X_test)
    
    print(f"\n✅ Meta features created!")
    print(f"   Train shape: {meta_train.shape}")
    print(f"   Test shape: {meta_test.shape}")
    
    return meta_train, meta_test


print("✅ Ensemble module loaded successfully!")
print("💡 Available: WeightedEnsemble, StackingEnsemble, Blending methods")