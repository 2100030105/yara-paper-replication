import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def generate_dataset():
    """
    Placeholder dataset (to be replaced with paper dataset later)
    """
    np.random.seed(42)
    X = np.random.rand(200, 20)   # 200 samples, 20 features
    y = np.random.randint(0, 2, 200)  # binary labels
    return X, y


def train_model(X_train, y_train):
    """
    Train baseline model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)


def main():
    print("Starting baseline pipeline...")

    # Step 1: Load / generate dataset
    X, y = generate_dataset()

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()