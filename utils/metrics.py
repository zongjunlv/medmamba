import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score,
    recall_score,
    roc_auc_score, 
    matthews_corrcoef,
    confusion_matrix
)

def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算准确率"""
    return float(accuracy_score(labels, preds))

def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """计算AUC，兼容二分类和多分类"""
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 2:
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return float(roc_auc_score(labels, probs[:, 1]))
            return float(roc_auc_score(labels, probs))
        return float(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))
    except ValueError:
        return 0.0


def compute_sensitivity_specificity(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    """计算敏感性(召回率)和特异性"""
    # 敏感性就是召回率(macro平均)
    sensitivity = float(recall_score(labels, preds, average='macro', zero_division=0))
    
    # 计算特异性 (对每个类别计算 TN/(TN+FP)，然后取macro平均)
    cm = confusion_matrix(labels, preds)
    n_classes = cm.shape[0]
    specificities = []
    
    for i in range(n_classes):
        # 对于类别i，TN是除了第i行和第i列之外的所有元素之和
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        # FP是第i列除了对角线元素的和
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    return sensitivity, float(np.mean(specificities))


def compute_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算macro F1分数"""
    return float(f1_score(labels, preds, average='macro', zero_division=0))


def compute_precision(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算macro Precision"""
    return float(precision_score(labels, preds, average='macro', zero_division=0))


def compute_matthews_corrcoef(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算Matthews相关系数 (多分类版本)"""
    try:
        return float(matthews_corrcoef(labels, preds))
    except:
        return 0.0



def compute_all_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray = None) -> dict:
    """计算所有指标"""
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = compute_accuracy(labels, preds)
    metrics['macro_f1'] = compute_macro_f1(labels, preds)
    metrics['precision'] = compute_precision(labels, preds)
    metrics['mcc'] = compute_matthews_corrcoef(labels, preds)
    
    # 敏感性和特异性
    sensitivity, specificity = compute_sensitivity_specificity(labels, preds)
    metrics['sensitivity'] = sensitivity
    metrics['specificity'] = specificity
    
    # AUC (如果提供了概率)
    if probs is not None:
        metrics['auc'] = compute_auc(labels, probs)
    
    return metrics
