"""
Interactive Visualization Functions with Plotly
เครื่องมือสำหรับสร้าง interactive plots ด้วย Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== FEATURE IMPORTANCE ====================

def plot_feature_importance(model, feature_names, top_n=20, 
                           title='Feature Importance', save_path=None):
    """
    Plot interactive feature importance
    
    Parameters:
    -----------
    model : trained model
        Model ที่มี feature_importances_ attribute
    feature_names : list
        ชื่อ features
    top_n : int
        จำนวน features ที่จะแสดง
    title : str
        Title ของ plot
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_feature_importance(model, X_train.columns, top_n=20)
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        print("⚠️  Model doesn't have feature importance!")
        return None
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)
    
    # Create plot
    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker=dict(
            color=importance_df['importance'],
            colorscale='Viridis',
            showscale=True
        ),
        text=importance_df['importance'].round(4),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#2c3e50')),
        xaxis_title='Importance',
        yaxis_title='Features',
        height=max(400, top_n * 25),
        template='plotly_white',
        hovermode='closest'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()
    
    return importance_df


def plot_feature_importance_comparison(models, feature_names, model_names=None,
                                      top_n=15, save_path=None):
    """
    เปรียบเทียบ feature importance จากหลาย models (interactive)
    
    Parameters:
    -----------
    models : list
        List of trained models
    feature_names : list
        ชื่อ features
    model_names : list, optional
        ชื่อของแต่ละ model
    top_n : int
        จำนวน features ที่จะแสดง
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_feature_importance_comparison(
    ...     [lgb_model, xgb_model],
    ...     X_train.columns,
    ...     model_names=['LightGBM', 'XGBoost']
    ... )
    """
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(models))]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=len(models),
        subplot_titles=model_names,
        horizontal_spacing=0.15
    )
    
    print(f"📊 Creating interactive feature importance comparison...")
    
    for idx, (model, name) in enumerate(zip(models, model_names), 1):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            print(f"⚠️  {name} doesn't have feature importance!")
            continue
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig.add_trace(
            go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                name=name,
                marker=dict(color=px.colors.qualitative.Set2[idx-1]),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ),
            row=1, col=idx
        )
    
    fig.update_layout(
        height=max(500, top_n * 30),
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Importance')
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()


# ==================== DISTRIBUTION PLOTS ====================

def plot_distributions(train_df, test_df, columns=None, save_path=None):
    """
    เปรียบเทียบ distribution ระหว่าง train และ test (interactive)
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    columns : list, optional
        Columns ที่ต้องการ plot
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_distributions(train, test, columns=['price', 'area'])
    """
    if columns is None:
        columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(2, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{col}' for col in columns],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    print(f"📊 Creating {len(columns)} interactive distributions...")
    
    col_iterator = tqdm(
        enumerate(columns),
        total=len(columns),
        desc="Creating plots",
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
    )
    
    for idx, col in col_iterator:
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        # Train histogram
        fig.add_trace(
            go.Histogram(
                x=train_df[col],
                name='Train',
                opacity=0.6,
                marker=dict(color='steelblue'),
                legendgroup='train',
                showlegend=(idx == 0),
                hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col_pos
        )
        
        # Test histogram
        fig.add_trace(
            go.Histogram(
                x=test_df[col],
                name='Test',
                opacity=0.6,
                marker=dict(color='coral'),
                legendgroup='test',
                showlegend=(idx == 0),
                hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=400 * n_rows,
        title_text='Train vs Test Distributions',
        template='plotly_white',
        barmode='overlay'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"\n✅ Saved to {save_path}")
    
    fig.show()


def plot_target_distribution(y, task='regression', title='Target Distribution', save_path=None):
    """
    Plot interactive target variable distribution
    
    Parameters:
    -----------
    y : pd.Series or np.array
        Target variable
    task : str
        'regression' หรือ 'classification'
    title : str
        Title ของ plot
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_target_distribution(y_train, task='regression')
    """
    if task == 'regression':
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Histogram', 'Box Plot'],
            specs=[[{'type': 'histogram'}, {'type': 'box'}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=y,
                marker=dict(color='steelblue'),
                name='Distribution',
                hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=y,
                marker=dict(color='steelblue'),
                name='Box Plot',
                hovertemplate='Value: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add stats annotation
        stats_text = f"Mean: {y.mean():.2f}<br>Median: {y.median():.2f}<br>Std: {y.std():.2f}"
        
    else:  # classification
        value_counts = pd.Series(y).value_counts().sort_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Count Plot', 'Pie Chart'],
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Bar plot
        fig.add_trace(
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker=dict(color='steelblue'),
                name='Count',
                hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=value_counts.index.astype(str),
                values=value_counts.values,
                hovertemplate='Class: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text=title,
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()


# ==================== CORRELATION ====================

def plot_correlation_heatmap(df, top_n=None, method='pearson', save_path=None):
    """
    Plot interactive correlation heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    top_n : int, optional
        แสดงเฉพาะ top_n features
    method : str
        'pearson', 'spearman', 'kendall'
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_correlation_heatmap(train, top_n=20)
    """
    print(f"📊 Computing {method} correlation...")
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)
    
    if top_n:
        top_features = corr.abs().sum().sort_values(ascending=False)[:top_n].index
        corr = numeric_df[top_features].corr(method=method)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Correlation Heatmap ({method.capitalize()})',
        height=max(600, len(corr) * 30),
        width=max(600, len(corr) * 30),
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()
    
    return corr


def plot_correlation_with_target(df, target_col, top_n=20, save_path=None):
    """
    Plot interactive correlation ของแต่ละ feature กับ target
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    target_col : str
        ชื่อ target column
    top_n : int
        จำนวน features ที่จะแสดง
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_correlation_with_target(train, 'price', top_n=20)
    """
    print(f"📊 Computing correlation with target...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df[numeric_cols].corrwith(df[target_col]).sort_values(ascending=True)
    correlations = correlations.drop(target_col)
    
    # Select top positive and negative
    top_positive = correlations.tail(top_n // 2)
    top_negative = correlations.head(top_n // 2)
    top_corr = pd.concat([top_negative, top_positive])
    
    # Create plot
    colors = ['red' if x < 0 else 'steelblue' for x in top_corr.values]
    
    fig = go.Figure(go.Bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation='h',
        marker=dict(color=colors),
        text=top_corr.values.round(3),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Correlation: %{x:.4f}<extra></extra>'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    
    fig.update_layout(
        title=f'Top {top_n} Features Correlated with {target_col}',
        xaxis_title='Correlation',
        yaxis_title='Features',
        height=max(500, top_n * 25),
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()
    
    return top_corr


# ==================== CLASSIFICATION METRICS ====================

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, save_path=None):
    """
    Plot interactive confusion matrix
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    labels : list, optional
        Class labels
    normalize : bool
        Normalize confusion matrix
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_confusion_matrix(y_val, y_pred, labels=['Class 0', 'Class 1'])
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_template = '%{z:.2%}'
    else:
        text_template = '%{z}'
    
    if labels is None:
        labels = [f'Class {i}' for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate=text_template,
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        width=500,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot interactive ROC curve
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_roc_curve(y_val, y_pred_proba)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='steelblue', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        width=700,
        template='plotly_white',
        hovermode='closest'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot interactive Precision-Recall curve
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_precision_recall_curve(y_val, y_pred_proba)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall curve',
        line=dict(color='steelblue', width=2),
        fill='tozeroy',
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=600,
        width=700,
        template='plotly_white',
        hovermode='closest'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()


# ==================== REGRESSION METRICS ====================

def plot_predictions(y_true, y_pred, save_path=None):
    """
    Plot interactive predictions vs actual values
    
    Parameters:
    -----------
    y_true : array
        True values
    y_pred : array
        Predicted values
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_predictions(y_val, y_pred)
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Predictions vs Actual', 'Residual Plot']
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            marker=dict(color='steelblue', size=5, opacity=0.6),
            name='Predictions',
            hovertemplate='True: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val, max_val = y_true.min(), y_true.max()
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect prediction',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Residuals plot
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            marker=dict(color='coral', size=5, opacity=0.6),
            name='Residuals',
            hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_xaxes(title_text='True Values', row=1, col=1)
    fig.update_yaxes(title_text='Predicted Values', row=1, col=1)
    fig.update_xaxes(title_text='Predicted Values', row=1, col=2)
    fig.update_yaxes(title_text='Residuals', row=1, col=2)
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()


# ==================== MISSING VALUES ====================

def plot_missing_values(df, save_path=None):
    """
    Interactive visualization of missing values
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    save_path : str, optional
        Path สำหรับบันทึก HTML
    
    Example:
    --------
    >>> plot_missing_values(train)
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✅ No missing values found!")
        return
    
    missing_pct = 100 * missing / len(df)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Missing Values Count', 'Missing Values Percentage']
    )
    
    # Count
    fig.add_trace(
        go.Bar(
            x=missing.values,
            y=missing.index,
            orientation='h',
            marker=dict(color='steelblue'),
            name='Count',
            hovertemplate='%{y}<br>Missing: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Percentage
    fig.add_trace(
        go.Bar(
            x=missing_pct.values,
            y=missing_pct.index,
            orientation='h',
            marker=dict(color='coral'),
            name='Percentage',
            hovertemplate='%{y}<br>Missing: %{x:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Count', row=1, col=1)
    fig.update_xaxes(title_text='Percentage (%)', row=1, col=2)
    
    fig.update_layout(
        height=max(400, len(missing) * 30),
        template='plotly_white',
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Saved to {save_path}")
    
    fig.show()
    
    print(f"\n📊 Summary:")
    print(f"   Total columns with missing: {len(missing)}")
    print(f"   Highest missing: {missing.index[0]} ({missing.values[0]} / {missing_pct.values[0]:.1f}%)")


print("✅ Plotly Visualization module loaded successfully!")
print("💡 All plots are interactive - zoom, pan, and hover!")
print("💡 Tip: Use save_path to save as HTML files")