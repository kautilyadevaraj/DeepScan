import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Load your embeddings (X) and labels (y)
X = np.load("../features/embeddings.npy")  # FaceNet embeddings
y = np.load("../features/labels.npy")  # Labels (real, fake, AI-generated)

# Ensure that the number of samples is greater than the perplexity value
n_samples = X.shape[0]
perplexity_value = min(30, n_samples - 1)  # Set perplexity less than number of samples

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.title("t-SNE Visualization of FaceNet Embeddings")
plt.colorbar(label='Class')

# Show the plot in a window
plt.show()
