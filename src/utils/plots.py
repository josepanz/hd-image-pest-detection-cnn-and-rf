import matplotlib.pyplot as plt

def plot_roc(fpr, tpr, auc, save_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_history(history, save_path):
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(save_path + "_loss.png")
    plt.close()

    plt.figure()
    plt.plot(history.history['auc'], label='train_auc')
    plt.plot(history.history['val_auc'], label='val_auc')
    plt.legend()
    plt.savefig(save_path + "_auc.png")
    plt.close()