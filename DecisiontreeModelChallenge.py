def import_tree_libs_wine_cv():
    """
    Imports necessary libraries for the Decision Tree classification project,
    including GridSearchCV and StratifiedKFold.

    Args:
        None

    Returns:
        dict: A dictionary containing references to the imported modules/functions.
    """
    # YOUR CODE HERE
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer

    # Return dictionary with references
    return {
        "np": np,
        "pd": pd,
        "plt": plt,
        "load_wine": load_wine,
        "train_test_split": train_test_split,
        "GridSearchCV": GridSearchCV,
        "StratifiedKFold": StratifiedKFold,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "plot_tree": plot_tree,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "make_scorer": make_scorer
    }

def load_wine_data():
    """
    Loads the Wine dataset from scikit-learn.

    Args:
        None

    Returns:
        sklearn.utils.Bunch: The loaded Wine dataset object, with data
                             structured as Pandas DataFrames/Series.
    """
    # We need the imported libraries first
    libs = import_tree_libs_wine_cv()
    load_wine = libs["load_wine"]
    # YOUR CODE HERE
    dataset= load_wine(as_frame=True)
    return dataset

def separate_wine_features_target(wine_bunch):
    """
    Separates the features (X) and target (y) from the loaded Wine dataset Bunch.

    Args:
        wine_bunch (sklearn.utils.Bunch): The loaded Wine dataset object with as_frame=True.

    Returns:
        tuple: A tuple containing:
               - X (pd.DataFrame): Features DataFrame.
               - y (pd.Series): Target Series.
    """
    # Get necessary library for type checks if needed
    libs = import_tree_libs_wine_cv()
    pd = libs["pd"]

    # YOUR CODE HERE
    X = wine_bunch.data
    y = wine_bunch.target
    return X, y

def tune_decision_tree(X, y):
    """
    Performs hyperparameter tuning for DecisionTreeClassifier using GridSearchCV
    with Stratified K-Fold Cross-Validation.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.

    Returns:
        sklearn.model_selection.GridSearchCV: The fitted GridSearchCV object.
    """
    # Get necessary classes
    libs = import_tree_libs_wine_cv()
    DecisionTreeClassifier = libs["DecisionTreeClassifier"]
    GridSearchCV = libs["GridSearchCV"]
    StratifiedKFold = libs["StratifiedKFold"]

    # YOUR CODE HERE
    dt_clf = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "ccp_alpha": [0.0, 0.001, 0.01, 0.1]
    }
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=dt_clf,
        param_grid=param_grid,
        cv= cv_strategy,
        scoring="accuracy",
        n_jobs=1
    )
    grid_search.fit(X, y)

    return grid_search

def get_best_tree_model(grid_search):
    """
    Extracts the best parameters, score, and model from a fitted GridSearchCV object.

    Args:
        grid_search (sklearn.model_selection.GridSearchCV): The fitted GridSearchCV object.

    Returns:
        dict: A dictionary containing the best parameters, score, and model.
              Keys: 'best_params', 'best_score', 'best_model'.
    """
    # Get necessary classes only if type checking is needed
    # libs = import_tree_libs_wine_cv()
    # DecisionTreeClassifier = libs["DecisionTreeClassifier"]

    # YOUR CODE HERE
    results ={
        "best_params": grid_search.best_params_,
        "best_score" : grid_search.best_score_,
        "best_model" : grid_search.best_estimator_
    }
    return results 

def plot_best_decision_tree(decision_tree_model, feature_names, class_names):
    """
    Visualizes the structure of a trained Decision Tree classifier.

    Args:
        decision_tree_model (DecisionTreeClassifier): The fitted tree model.
        feature_names (list): List of feature names.
        class_names (list): List of target class names.

    Returns:
        list: A list of Matplotlib artists drawn by plot_tree.
    """
    # Get necessary libraries/functions
    libs = import_tree_libs_wine_cv()
    plt = libs["plt"]
    plot_tree = libs["plot_tree"]

    # YOUR CODE HERE
    fig, ax = plt.subplots(figsize=(20, 10))

    tree_artists = plot_tree(
        decision_tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    ax.set_title("Best Decision Tree Structure (Wine Dataset)")
    plt.show()
    return tree_artists