---
name: data-mining
description: Data Mining. Unsupervised pattern discovery - clustering and customer segmentation (k-means, k-medoids, hierarchical, DBSCAN, HDBSCAN, Gaussian mixtures), association rules and market basket analysis (Apriori, FP-Growth, support, confidence, lift), anomaly and outlier detection (Isolation Forest, LOF, one-class SVM, robust z-scores, autoencoders), dimensionality reduction for exploration (PCA, t-SNE, UMAP), sequence mining, recommenders and text mining. Use for segmentation, clustering, unsupervised learning, basket analysis, fraud or anomaly screening, topic discovery.
---

# Data Mining

Unsupervised results are only as good as the decision they support. Start with what each segment, rule or alert will trigger.

## Clustering

1. Choose features that matter for the action; scale them inside a pipeline (`StandardScaler` or `RobustScaler`) before any distance method.
2. Compare k (or density parameters) with several lenses: elbow (inertia), silhouette, gap statistic, BIC for mixtures, and stability (adjusted Rand index across bootstrap refits). Silhouette favors convex blobs and misleads on non-convex shapes; DBSCAN or HDBSCAN suit those.
3. Profile every cluster: size, share, defining features (mean or median versus overall, standardized difference), and a plain-language name grounded in those numbers.
4. Validate usefulness: outcome rates by cluster on held-out data, or an A/B test of the cluster-specific action.

## Association rules

Report support, confidence, lift, leverage and conviction for each rule. Lift above 1 means items co-occur more than chance; below 1 means they repel. Set a minimum support that yields at least tens of transactions per rule; low-support rules are often spurious. Prune redundant rules and check stability across time windows.

## Anomaly detection

State the expected contamination and the review process (who looks at alerts, how many per day). Methods: robust z-score (median and MAD) per feature, Isolation Forest, local outlier factor, one-class SVM, autoencoder reconstruction error. With few labels, evaluate precision at k (k = review capacity) on labelled samples, and monitor alert volume over time.

## Dimensionality reduction for exploration

PCA for linear structure with explained variance and loadings. t-SNE and UMAP for visual neighborhoods only: their distances and cluster sizes are not metric-faithful and must not drive quantitative claims or clustering inputs.

## Recommenders, sequences, text

Collaborative filtering with implicit feedback (ALS, item-item cosine), popularity baseline, cold-start fallback; evaluate with time-based holdout and precision at k, recall at k, NDCG, coverage. Sequence mining (PrefixSpan) with minimum support. Text: TF-IDF plus linear models first, topic models (NMF, LDA) with coherence, embeddings for semantic similarity.

Methods, parameters and code: `references/mining-methods.md`.
