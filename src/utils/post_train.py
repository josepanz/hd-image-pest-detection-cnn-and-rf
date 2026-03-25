from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

def evaluate_model(model, X_val, y_val, threshold, auc_path):
    
    y_probs = model.predict(X_val).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n--- REPORTE DE CLASIFICACIÓN ---")
    print(classification_report(y_val, y_pred))

    # ROC
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\nAUC REAL: {roc_auc:.4f}")

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid()
    plt.savefig(auc_path)
    plt.show()