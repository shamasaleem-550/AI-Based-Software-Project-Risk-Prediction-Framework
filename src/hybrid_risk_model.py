import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_hybrid_model():
    # Load combined data
    df = pd.read_csv("results/combined_risk_data.csv")

    X = df[['ambiguity_score', 'overload_score']]
    y = df['risk_level']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions & metrics
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save trained model
    joblib.dump(clf, 'results/hybrid_risk_model.pkl')
    print("Hybrid risk model saved to results/hybrid_risk_model.pkl")

    return clf
