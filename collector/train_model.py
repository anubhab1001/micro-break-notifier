# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

CSV_FILE = "user_activity_log.csv"
MODEL_FILE = "stress_rf_model.pkl"

EMO_COLS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FEATURE_COLS = [
    "BlinkRate", "EyeRatio", "MouthRatio",
    "AvgTypingInterval", "MaxPause", "AvgMouseIdle",
    *[f"emo_{c}" for c in EMO_COLS]
]

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    # keep rows with a label 0/1/2
    df = df[df["StressLevel"].isin([0,1,2])]
    # ensure all required feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    # fill any remaining NaNs
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df

def main():
    if not os.path.exists(CSV_FILE):
        print("CSV not found. Run the collector first.")
        return

    df = load_and_clean(CSV_FILE)
    if len(df) < 30:
        print("Not enough samples to train. Collect more (>= 30 rows).")
        return

    X = df[FEATURE_COLS].astype(float)
    y = df["StressLevel"].astype(int)

    # train/val split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # class weights for imbalance
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {cls: w for cls, w in zip(classes, weights)}
    print("Class weights:", class_weight)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\n=== Evaluation ===")
    print(classification_report(y_test, preds, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # feature importances
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:")
    print(importances.head(12))

    # persist
    joblib.dump(clf, MODEL_FILE)
    print(f"\nSaved model to {MODEL_FILE}")
    # store columns used (so live predictor can align)
    # joblib already stores feature_names_in_, but we can be explicit:
    clf.feature_names_in_ = np.array(FEATURE_COLS)
    joblib.dump(clf, MODEL_FILE)

if __name__ == "__main__":
    main()
