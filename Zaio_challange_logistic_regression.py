def import_classification_libraries():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        ConfusionMatrixDisplay
    )

    return {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "datasets": datasets,
        "train_test_split": train_test_split,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "classification_report": classification_report,
        "ConfusionMatrixDisplay": ConfusionMatrixDisplay
    }


def load_cancer_data():
    libs = import_classification_libraries()
    datasets = libs["datasets"]
    return datasets.load_breast_cancer(as_frame=True)


def explore_cancer_data(cancer_bunch):
    libs = import_classification_libraries()
    pd = libs["pd"]

    df = cancer_bunch.frame
    target_distribution = df["target"].value_counts()
    missing_values_sum = df.isnull().sum()
    description = df.describe()

    return {
        "dataframe": df,
        "target_distribution": target_distribution,
        "missing_values_sum": missing_values_sum,
        "description": description
    }


def prepare_cancer_data(cancer_bunch):
    libs = import_classification_libraries()
    train_test_split = libs["train_test_split"]

    X = cancer_bunch.data
    y = cancer_bunch.target

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_logistic_regression_model(X_train, y_train):
    libs = import_classification_libraries()
    LogisticRegression = libs["LogisticRegression"]

    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test):
    libs = import_classification_libraries()
    accuracy_score = libs["accuracy_score"]
    confusion_matrix = libs["confusion_matrix"]
    precision_score = libs["precision_score"]
    recall_score = libs["recall_score"]
    f1_score = libs["f1_score"]
    classification_report = libs["classification_report"]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    specificity_class0 = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision_class1 = precision_score(y_test, y_pred, pos_label=1)
    recall_class1 = recall_score(y_test, y_pred, pos_label=1)
    f1_class1 = f1_score(y_test, y_pred, pos_label=1)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Malignant (0)", "Benign (1)"]
    )

    return {
        "predictions": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "precision_class1": precision_class1,
        "recall_class1": recall_class1,
        "specificity_class0": specificity_class0,
        "f1_score_class1": f1_class1,
        "classification_report": report
    }


def plot_confusion_matrix(model, X_test, y_test, display_labels):
    libs = import_classification_libraries()
    plt = libs["plt"]
    ConfusionMatrixDisplay = libs["ConfusionMatrixDisplay"]

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=display_labels,
        cmap=plt.cm.Blues,
        values_format='d'
    )

    ax = disp.ax_
    ax.set_title("Confusion Matrix")
    return ax


# Run the full pipeline
if __name__ == "__main__":
    libs = import_classification_libraries()
    plt = libs["plt"]

    # Load and prepare data
    cancer_data = load_cancer_data()
    X_train, X_test, y_train, y_test = prepare_cancer_data(cancer_data)

    # Train model
    model = train_logistic_regression_model(X_train, y_train)

    # Evaluate model
    results = evaluate_classification_model(model, X_test, y_test)
    print("Accuracy:", results["accuracy"])
    print("Confusion Matrix:\n", results["confusion_matrix"])
    print("Classification Report:\n", results["classification_report"])

    # Plot confusion matrix
    ax = plot_confusion_matrix(model, X_test, y_test, display_labels=["Malignant", "Benign"])
    plt.show()
