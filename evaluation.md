1. Good Tests for Comparison
Retrieval Accuracy:
Use text-to-image and image-to-text retrieval tasks. For each query, rank the database and measure how often the correct match is in the top-k results.
Clustering Quality:
Cluster embeddings (e.g., with k-means) and measure how well clusters correspond to known categories (purity, NMI).
Visualization:
Use UMAP/t-SNE to visualize embedding spaces and check for separation of classes or concepts.
2. Recommended Metrics
Recall@K / Precision@K:
For retrieval, how often is the correct item in the top-k results?
Mean Reciprocal Rank (MRR):
Average of reciprocal ranks of the correct answer.
Normalized Mutual Information (NMI):
For clustering, measures alignment between clusters and ground truth.
Silhouette Score:
Measures how well embeddings are separated into clusters.
3. Measuring Disentanglement
Factor Correlation:
Measure how independent different embedding dimensions are (low correlation = more disentangled).
Mutual Information Gap (MIG):
Quantifies how well each latent dimension captures a single factor of variation.
Linear Probing:
Train simple classifiers on individual embedding dimensions to predict factors; high accuracy on single dimensions suggests disentanglement.
Intervention Tests:
Manipulate one factor (e.g., change text, keep image) and see if only the relevant embedding changes.
4. Practical Steps
Run retrieval tasks with both models and compare Recall@K, MRR.
Visualize both embedding spaces and compare cluster separation.
Compute correlation matrices for embedding dimensions.
Optionally, use disentanglement metrics from the literature (e.g., MIG).
Summary:
Use retrieval metrics (Recall@K, MRR), clustering metrics (NMI, silhouette), and disentanglement metrics (correlation, MIG, probing).
A good test is text-to-image retrieval and visualizing the embedding space for both models.
Disentanglement is best measured by independence of embedding dimensions and their alignment with interpretable factors.