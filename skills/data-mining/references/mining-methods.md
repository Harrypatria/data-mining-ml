# Mining methods

## Clustering algorithms

| Algorithm | Shape assumption | Key parameters | Scale needed | Notes |
|---|---|---|---|---|
| k-means | Convex, similar size | k, n_init | Yes | Fast; sensitive to outliers |
| k-medoids | Convex, robust | k, metric | Yes | Medoids are real rows, easier to explain |
| Hierarchical (Ward) | Nested groups | linkage, distance threshold | Yes | Dendrogram shows structure; O(n^2) memory |
| DBSCAN | Arbitrary, dense | eps, min_samples | Yes | Labels noise as -1; eps from k-distance plot |
| HDBSCAN | Arbitrary, varying density | min_cluster_size | Yes | Fewer parameters than DBSCAN |
| Gaussian mixture | Ellipsoidal, soft | n_components, covariance_type | Yes | BIC for model choice; probabilities per row |

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rows = []
for k in range(2, 9):
    pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=SEED)).fit(X)
    labels = pipe[-1].labels_
    boots = []
    for b in range(10):
        idx = rng.choice(len(X), len(X))
        refit = make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=b)).fit(X[idx])
        boots.append(adjusted_rand_score(labels, refit.predict(X)))
    rows.append({"k": k, "inertia": pipe[-1].inertia_, "silhouette": silhouette_score(pipe[:-1].transform(X), labels),
                 "stability_ari": np.mean(boots)})
```

Gap statistic: compare log inertia with its expectation under uniform reference data drawn in the bounding box (10 references), choose the smallest k with gap(k) >= gap(k+1) - s(k+1).

Cluster profile table: per cluster `size`, `share`, then for each feature the cluster median and the standardized difference from the overall mean; sort features by absolute difference and name the cluster from the top two.

## Association rules

```python
from mlxtend.frequent_patterns import association_rules, fpgrowth
itemsets = fpgrowth(basket_onehot, min_support=0.01, use_colnames=True)
rules = association_rules(itemsets, metric="lift", min_threshold=1.2)
rules = rules[(rules["support"] * n_transactions >= 30)].sort_values("lift", ascending=False)
```

| Metric | Formula | Reads as |
|---|---|---|
| Support | P(A and B) | How common the pair is |
| Confidence | P(B given A) | How often B follows A |
| Lift | confidence / P(B) | Above 1 attraction, below 1 repulsion |
| Leverage | P(A and B) - P(A)P(B) | Absolute co-occurrence excess |
| Conviction | (1 - P(B)) / (1 - confidence) | How much A implies B; infinite for perfect rules |

## Anomaly detection

| Method | Good for | Parameter |
|---|---|---|
| Robust z-score | Single features, explainable | threshold 3.5 on (x - median) / (1.4826 x MAD) |
| Isolation Forest | Mixed tabular | contamination equal to expected alert rate |
| Local outlier factor | Local density differences | n_neighbors 20 to 50 |
| One-class SVM | Small, clean training set of normal data | nu, gamma |
| Autoencoder | High-dimensional, non-linear | Threshold on validation reconstruction error percentile |

Evaluation with scarce labels: precision at k, recall among known incidents, alert volume per day, analyst agreement rate.

## Dimensionality reduction

PCA: report cumulative explained variance and top loadings per component. t-SNE (`perplexity` 30) and UMAP (`n_neighbors` 15, `min_dist` 0.1) are for pictures; never cluster on their output or compare distances across the map.

## Recommenders

Baselines: global popularity, popularity within segment. Implicit ALS (confidence weighting), item-item cosine on co-occurrence. Cold start: content features or popularity fallback. Evaluate on the last interactions per user (time-based), with precision at k, recall at k, NDCG at k, catalog coverage and novelty.

## Text mining

Pipeline: normalize, TF-IDF (word 1-2 grams, min_df), linear classifier baseline; NMF or LDA topics with coherence and top terms; sentence embeddings for semantic search and clustering. Deduplicate near-identical texts before splitting.

## SQL and large data

Push aggregation into the warehouse (GROUP BY per entity, window functions for lags), pull only feature tables. DuckDB for local Parquet, Polars lazy frames for out-of-core pipelines, Spark MLlib (KMeans, FPGrowth, BisectingKMeans) for cluster-scale data.
