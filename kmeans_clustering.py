from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    X = np.array([
        [1, 2], [1, 4], [1, 0],
        [10, 2], [10, 4], [10, 0]
    ])

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    print("Centroids:", kmeans.cluster_centers_)

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='X')
    plt.show()

if __name__ == "__main__":
    main()
