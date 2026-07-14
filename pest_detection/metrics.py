from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

def evaluar_modelo(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    return cm, report, auc, fpr, tpr