import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generate_synthetic_data(n_samples, centers, cluster_std, random_state):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        n_features=2
    )
    return X, y

def plot_raw_data(X):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6, c='blue', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Raw Generated Data Before Clustering")
    plt.grid(True)
    return plt.gca()

def calculate_wcss(X, k_range):
    wcss_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    return wcss_values

def plot_elbow_method(k_range, wcss_values):
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), wcss_values, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(list(k_range))
    plt.grid(True)
    return plt.gca()

def calculate_silhouette_scores(X, k_range):
    silhouette_avg_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_avg_scores.append(score)
    return silhouette_avg_scores

def plot_silhouette_scores(k_range, silhouette_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), silhouette_scores, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Score for Optimal K")
    plt.xticks(list(k_range))
    plt.grid(True)
    return plt.gca()

def train_final_kmeans(X, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
    kmeans.fit(X)
    return kmeans

def plot_clustered_data(X, labels, centers):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=200,
        c="red",
        marker="X",
        edgecolors="k",
        label="Centroids"
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"K-Means Clustering Results (K={len(centers)})")
    plt.grid(True)
    plt.legend()
    return plt.gca()

if __name__ == "__main__":
    N_SAMPLES = 300
    CENTERS = 4
    CLUSTER_STD = 0.70
    RANDOM_STATE = 42

    X, y_true = generate_synthetic_data(N_SAMPLES, CENTERS, CLUSTER_STD, RANDOM_STATE)
    plot_raw_data(X)
    plt.show()

    k_values = range(1, 11)
    wcss = calculate_wcss(X, k_values)
    plot_elbow_method(k_values, wcss)
    plt.show()

    silhouette_range = range(2, 11)
    silhouette_scores = calculate_silhouette_scores(X, silhouette_range)
    plot_silhouette_scores(silhouette_range, silhouette_scores)
    plt.show()

    optimal_k = 4
    final_model = train_final_kmeans(X, optimal_k)
    plot_clustered_data(X, final_model.labels_, final_model.cluster_centers_)
    plt.show()
