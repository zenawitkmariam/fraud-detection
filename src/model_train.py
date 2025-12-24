from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, average_precision_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1)
}

def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    print(f"\n==== Results for {dataset_name} ====")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # We use Average Precision (AUPRC) because it's better for imbalanced data
        auprc = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        print(f"{name}: F1-Score = {f1:.4f} | AUPRC = {auprc:.4f}")
