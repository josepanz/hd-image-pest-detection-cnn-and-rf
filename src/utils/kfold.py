from sklearn.model_selection import StratifiedKFold
import numpy as np

def run_kfold(model_builder, X, y, folds=5):

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔥 Fold {fold+1}/{folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_builder()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=16,
            verbose=1
        )

        y_pred = model.predict(X_val)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, y_pred)

        print(f"AUC Fold {fold+1}: {auc:.4f}")
        aucs.append(auc)

    print("\n📊 RESULTADO FINAL K-FOLD")
    print(f"AUC promedio: {np.mean(aucs):.4f}")
    print(f"Std: {np.std(aucs):.4f}")

    return aucs