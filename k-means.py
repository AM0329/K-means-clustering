# K-Means Clustering Algorithm on Random Data with Two Features

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Generating random data
X = np.random.random((50, 2))
print("Generated Data:\n", X)

# Scatter plot for original data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Selecting two random centroids from data
random_index = random.sample(range(0, X.shape[0]), 2)
centroids = X[random_index]
old_centroids = centroids
print("Initial Centroids:\n", centroids)

# K-Means clustering algorithm
for i in range(200):
    distances = []
    group = []
    new_centroids = []

    # Computing Euclidean distance between the centroids and all points
    for row in X:
        for centroid in centroids:
            distances.append(np.sqrt(np.sum(np.power(row - centroid, 2), keepdims=True)))
        min_dist = min(distances)
        index_position = distances.index(min_dist)
        group.append(index_position)  # Making a group of index having minimum distance
        distances.clear()

    # Computing the new centroids
    cluster_type = np.unique(group)
    for types in cluster_type:
        new_centroids.append(X[np.array(group) == types].mean(axis=0))
    new_centroids = np.array(new_centroids)
    print("New Centroids:\n", new_centroids)

    # Checking whether old centroids are not equal to new centroids, if yes repeat the above steps else stop
    if np.not_equal(centroids, new_centroids).all():
        centroids = new_centroids  # Moving the centroids
    else:
        break

final = np.array(group)
print("Cluster Groups:\n", final)
centers = np.array(new_centroids)
print("Final Centroids:\n", centers)

# After K-Means clustering, scatter plot of dataset
classes = ['First Cluster', 'Second Cluster', 'Centroids']
redClass = X[final == 0]
blueClass = X[final == 1]

plt.scatter(redClass[:, 0], redClass[:, 1], c='r', label=classes[0])
plt.scatter(blueClass[:, 0], blueClass[:, 1], c='b', label=classes[1])
plt.scatter(centers[:, 0], centers[:, 1], c='green', label=classes[2], edgecolor='black')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper right')
plt.show()
