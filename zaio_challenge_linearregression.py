def import_libraries():
    """
    Imports necessary libraries for the project.

    Returns:
        dict: A dictionary containing references to the imported modules/functions.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    return {
        "pd": pd,
        "np": np,
        "plt": plt,
        "datasets": datasets,
        "train_test_split": train_test_split,
        "LinearRegression": LinearRegression,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
    }


def load_diabetes_data():
    """Loads the diabetes dataset as a Pandas DataFrame inside a Bunch object."""
    libs = import_libraries()
    datasets = libs["datasets"]
    diabetes = datasets.load_diabetes(return_X_y=False, as_frame=True)
    return diabetes


def prepare_data(diabetes_bunch, feature_name):
    """Select one feature and split into train/test sets."""
    libs = import_libraries()
    train_test_split = libs["train_test_split"]

    X = diabetes_bunch.frame[[feature_name]]
    y = diabetes_bunch.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_linear_regression_model(X_train, y_train):
    """Train a Linear Regression model."""
    libs = import_libraries()
    LinearRegression = libs["LinearRegression"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with MSE and R²."""
    libs = import_libraries()
    mean_squared_error = libs["mean_squared_error"]
    r2_score = libs["r2_score"]

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"predictions": y_pred, "mse": mse, "r2_score": r2}


def plot_regression_results(X_test, y_test, y_pred, feature_name):
    """Plot actual test data and regression line."""
    libs = import_libraries()
    plt = libs["plt"]

    plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.6)
    plt.plot(X_test, y_pred, color="red", label="Regression Line", linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel("Diabetes Progression")
    plt.title(f"Diabetes Progression vs. {feature_name}")
    plt.legend()
    plt.show()

    return plt.gca()


# Run everything in VS Code
if __name__ == "__main__":
    diabetes = load_diabetes_data()
    X_train, X_test, y_train, y_test = prepare_data(diabetes, "bmi")

    model = train_linear_regression_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    print("Model Evaluation:")
    print("MSE:", results["mse"])
    print("R² Score:", results["r2_score"])

    plot_regression_results(X_test, y_test, results["predictions"], "bmi")
