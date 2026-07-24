# an Unsupervised learning project:
# PCA, K-means, and t-SNE on handwritten digits, using the sklearn digits dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

# loading data
digits = load_digits()

X = digits.data
y = digits.target # since it's unsupervised, we don't use y for training, but only later
# to see if the predictions are correct

# X.shape = (1797, 64)
# y.shape = (1797,)
# 1797 images, 64 features per image (each image is 8 x 8 pixels)

# ------------------------------------------------------------

# 1. PCA (reducing dimonsionality, from 64 features down to 2)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X) # scaling the data, because pca needs it

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print("Total variance explained by 2 components:")
print(pca.explained_variance_ratio_.sum()) # 21%, so the 2 components keep 21% of info


plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="tab10",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA visualization of handwritten digits")
plt.colorbar(scatter, label="True digit label")
plt.show()

# PCA shape: (1797, 2)
# Total variance explained by 2 components: 0.21594970500832794

# ------------------------------------------------------------

# 2. K-means (clustering)

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

c = 10 # choosing 10 clusters since there are 10 digits: 0, 1, 2, ..., 9.
kmeans = KMeans(
    n_clusters=c,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

print("\nNumber of samples in each cluster:")
print(np.bincount(clusters))

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-means clusters visualized with PCA")

plt.colorbar(scatter, label="K-means cluster")
plt.show()


cluster_table = pd.crosstab( # comparing with true digit labels
    pd.Series(clusters, name="K-means cluster"),
    pd.Series(y, name="True digit")
)

print("\nCluster vs true digit table:")
print(cluster_table)


ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)

print(f"\nAdjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")

# Adjusted Rand Index: 0.5344 (moderate clustering quality. 0 -> random, 1 -> perfect)
# Normalized MI: 0.6712 (decent relationship between clusters-true digit labels.)

# ------------------------------------------------------------

# 3. t-SNE (visualization)

from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

print("t-SNE shape:", X_tsne.shape)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="tab10",
    alpha=0.7
)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE visualization of handwritten digits")
plt.colorbar(scatter, label="True digit label")
plt.show()

# t-SNE shape: (1797, 2)
# cleaner-looking groups than PCA
# PCA tries to keep global variance, t-SNE tries to keep local neighborhoods
