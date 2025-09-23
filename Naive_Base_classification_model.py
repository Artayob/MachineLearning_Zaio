def import_bayes_libraries():
    """
    Imports necessary libraries for the Naive Bayes text classification project.

    Args:
        None

    Returns:
        dict: A dictionary containing references to the imported modules/functions.
    """
    # YOUR CODE HERE
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        ConfusionMatrixDisplay
    )

    return {
        "np": np,
        "plt": plt,
        "sns": sns,
        "fetch_20newsgroups": fetch_20newsgroups,
        "train_test_split": train_test_split,
        "TfidfVectorizer": TfidfVectorizer,
        "MultinomialNB": MultinomialNB,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "ConfusionMatrixDisplay": ConfusionMatrixDisplay
    }

def load_newsgroup_data(categories_to_load):
    """
    Loads a specified subset of the 20 Newsgroups dataset.

    Args:
        categories_to_load (list): A list of category names (strings) to load.

    Returns:
        sklearn.utils.Bunch: The loaded dataset object containing data, target,
                             target_names, etc.
    """
    # Get necessary function
    libs = import_bayes_libraries()
    fetch_20newsgroups = libs["fetch_20newsgroups"]
    np = libs["np"] # For type checks if needed

    # YOUR CODE HERE
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories_to_load,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes")
    )

    return dataset

def prepare_text_data(news_bunch):
    """
    Performs TF-IDF vectorization on the text data and splits it into
    training and testing sets.

    Args:
        news_bunch (sklearn.utils.Bunch): The loaded newsgroup dataset object.

    Returns:
        tuple: A tuple containing:
               - tfidf_vectorizer (TfidfVectorizer): The fitted vectorizer.
               - X_train_tfidf (sparse matrix): TF-IDF features for training.
               - X_test_tfidf (sparse matrix): TF-IDF features for testing.
               - y_train (np.ndarray): Training target labels.
               - y_test (np.ndarray): Testing target labels.
    """
    # Get necessary classes/functions
    libs = import_bayes_libraries()
    TfidfVectorizer = libs["TfidfVectorizer"]
    train_test_split = libs["train_test_split"]
    np = libs["np"] # For type checks if needed

    # YOUR CODE HERE
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf_vectorizer.fit_transform(news_bunch.data)
    y = news_bunch.target
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.25, random_state=42
    )

    return tfidf_vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test

def train_naive_bayes_model(X_train, y_train):
    """
    Creates and trains a Multinomial Naive Bayes model.

    Args:
        X_train (sparse matrix): Training data features (TF-IDF).
        y_train (np.ndarray): Training data target labels.

    Returns:
        sklearn.naive_bayes.MultinomialNB: The trained MultinomialNB model instance.
    """
    # Get the MultinomialNB class
    libs = import_bayes_libraries()
    MultinomialNB = libs["MultinomialNB"]
    np = libs["np"] # For type checks if needed

    # YOUR CODE HERE
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    return nb_model

def evaluate_naive_bayes_model(model, X_test, y_test, target_names):
    """
    Evaluates the trained Naive Bayes model on the test set.

    Args:
        model (sklearn.naive_bayes.MultinomialNB): The trained model.
        X_test (sparse matrix): Test data features (TF-IDF).
        y_test (np.ndarray): Actual test data target labels.
        target_names (list): List of category names for the report.

    Returns:
        dict: A dictionary containing predictions and evaluation metrics.
              Keys: 'predictions', 'accuracy', 'confusion_matrix',
                    'classification_report'.
    """
    # Get evaluation metric functions
    libs = import_bayes_libraries()
    np = libs["np"]
    accuracy_score = libs["accuracy_score"]
    confusion_matrix = libs["confusion_matrix"]
    classification_report = libs["classification_report"]

    # YOUR CODE HERE
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=target_names)

    return {
        "predictions": y_pred,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": cr
    }

def plot_bayes_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix using a Seaborn heatmap.

    Args:
        cm (np.ndarray): The confusion matrix (e.g., from evaluate_naive_bayes_model).
        class_names (list): List of strings for class labels.

    Returns:
        matplotlib.axes._axes.Axes: The Axes object containing the heatmap plot.
    """
    # Get plotting libraries
    libs = import_bayes_libraries()
    plt = libs["plt"]
    sns = libs["sns"]
    np = libs["np"] # For type checks if needed

    # YOUR CODE HERE
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel("Actual Category")
    plt.xlabel("Predicted Category")
    plt.title("Naive Bayes Confusion Matrix")
    plt.tight_layout()

    return ax

if __name__ == "__main__":
    categories = [
        "comp.graphics",
        "rec.sport.baseball",
        "sci.med",
        "talk.politics.misc"
    ]

    # Step 2: Load data
    dataset = load_newsgroup_data(categories)

    # Step 3: Prepare data (vectorization + split)
    tfidf_vectorizer, X_train, X_test, y_train, y_test = prepare_text_data(dataset)

    # Step 4: Train model
    model = train_naive_bayes_model(X_train, y_train)

    # Step 5: Evaluate model
    results = evaluate_naive_bayes_model(model, X_test, y_test, dataset.target_names)
    print("✅ Accuracy:", results["accuracy"] *100)
    print("\n📊 Classification Report:\n", results["classification_report"])

    # Step 6: Plot confusion matrix
    ax = plot_bayes_confusion_matrix(results["confusion_matrix"], dataset.target_names)
    import matplotlib.pyplot as plt
    plt.show()
