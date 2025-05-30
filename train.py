#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SageMaker Training Script for Fraud Detection using Graph-Based Community Detection.

This script performs the following steps:
1. Loads credit card transaction data.
2. Performs feature scaling and DBSCAN clustering on transaction features (V1-V28).
3. Identifies fraudulent clusters based on the proportion of fraudulent transactions within them.
4. Constructs a graph where nodes are clusters and edges represent similarity between clusters.
5. Applies the Louvain algorithm (PLM) for community detection on this graph.
6. Analyzes the fraud concentration within detected communities.
7. Saves model artifacts: scaler, cluster labels, scaled features, and community assignments.

Environment Variables:
- SM_MODEL_DIR: (Set by SageMaker) Directory to save model artifacts. Defaults to '.' for local runs.
- SM_CHANNEL_TRAINING: (Set by SageMaker) Directory where training data is located. This script currently
  assumes 'creditcard.csv' is in the same directory as the script or in a path accessible to SageMaker.

Usage:
- Local: python train.py
- SageMaker: The script is executed by the SageMaker training job.
"""
import argparse
import os
import sys

# --- Environment Setup for Parallelism ---
# Set environment variables to control the number of threads used by numerical libraries.
# This helps in optimizing performance and ensuring consistent behavior across different environments.
# It's particularly important for libraries like NumPy (OpenBLAS/MKL), SciPy, and Networkit.
n_cores = (
    os.cpu_count() or 1
)  # Get the number of CPU cores, default to 1 if not detectable.
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)

import torch
import networkit as nk
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Explicitly set the number of threads for PyTorch and Networkit.
# This complements the environment variable settings.
torch.set_num_threads(n_cores)
nk.setNumberOfThreads(n_cores)
print(f"Configured to use {n_cores} cores/threads for numerical libraries.")


def main():
    """
    Main function to execute the training pipeline.

    Handles data loading, preprocessing, clustering, graph construction,
    community detection, fraud analysis, and saving of model artifacts.
    """
    # --- Configuration & Setup ---
    # SM_MODEL_DIR: Environment variable set by SageMaker. It specifies the directory
    # where the model artifacts (trained model, scaler, etc.) should be saved.
    # Defaults to the current directory (".") if not set (e.g., for local execution outside SageMaker).
    model_dir = os.environ.get("SM_MODEL_DIR", ".")
    # Create the model directory if it doesn't exist, especially for local runs.
    os.makedirs(model_dir, exist_ok=True)
    print("Starting SageMaker training script.")
    print(f"Output directory for model artifacts: {model_dir}")

    # Print library versions for reproducibility and debugging.
    print(f"PyTorch version: {torch.__version__}")
    print(f"Networkit version: {nk.__version__}")

    # Determine the computation device (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU).
    # MPS (Metal Performance Shaders) is preferred on compatible Macs.
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --- Data Loading ---
    # data_path: Path to the credit card fraud dataset.
    # In a SageMaker environment, data is typically provided via channels.
    # SM_CHANNEL_TRAINING would point to the directory containing 'creditcard.csv'.
    # For simplicity in this example, we check for 'creditcard.csv' in the input data directory
    # or the script's current directory for local runs.
    sagemaker_input_data_dir = os.environ.get("SM_CHANNEL_TRAINING", ".")
    data_path = os.path.join(sagemaker_input_data_dir, "creditcard.csv")
    print(f"Attempting to load data from: {data_path}")
    if not os.path.exists(data_path):
        # If the dataset is not found, raise an error with clear instructions.
        raise FileNotFoundError(
            f"Dataset {data_path} not found. Please download from Kaggle (or ensure it's in the correct path for SageMaker) and place in the working directory."
        )

    # For local development (identified if SM_MODEL_DIR is not set or is '.'), load a subset of data to speed up testing.
    # In a SageMaker environment (where SM_MODEL_DIR is typically '/opt/ml/model'), the full dataset is loaded.
    # This check can also be based on other SageMaker environment variables if preferred.
    is_local_run = (
        os.environ.get("SM_MODEL_DIR") == "." or "SM_MODEL_DIR" not in os.environ
    )
    if is_local_run:
        print("Running in local mode, loading a subset of data (5000 rows).")
        df = pd.read_csv(data_path, nrows=5000)
    else:
        print("Running in SageMaker mode (or equivalent), loading full dataset.")
        df = pd.read_csv(data_path)
    print(
        f"Loaded dataset '{data_path}': {df.shape[0]} transactions, {df.shape[1]} features."
    )

    # --- Feature Engineering & Clustering ---
    # Extract PCA components (V1-V28) as features for clustering.
    # These features are anonymized and are the result of a PCA transformation.
    features = df[[f"V{i}" for i in range(1, 29)]].values

    # Scale features using StandardScaler to have zero mean and unit variance.
    # This is important for distance-based algorithms like DBSCAN.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Save the scaled features along with original index for later mapping or analysis.
    # This helps in associating clustered transactions back to their original entries.
    # The 'index' column here refers to the original DataFrame's index, useful for traceability.
    features_df = pd.DataFrame(features_scaled, columns=[f"V{i}" for i in range(1, 29)])
    features_df.insert(
        0, "index", df.index
    )  # Add original DataFrame index as the first column.
    features_output_path = os.path.join(
        model_dir, "creditcard_features.csv"
    )  # Renamed for clarity in serve.py
    features_df.to_csv(features_output_path, index=False)
    print(f"Saved scaled features to {features_output_path}")

    # Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    # These parameters (eps=2.5, min_samples=10) are based on the script's original configuration.
    # Transactions are clustered based on their V1-V28 feature similarity.
    print("Applying DBSCAN clustering with original parameters (eps=2.5, min_samples=10)...")
    dbscan = DBSCAN(eps=2.5, min_samples=10)  # Reverted to original parameters
    cluster_labels = dbscan.fit_predict(features_scaled) # Returns cluster labels for each point. Noise points are labeled -1.
    df["cluster"] = cluster_labels  # Add cluster labels to the DataFrame
    print(
        f"DBSCAN found {len(np.unique(cluster_labels))} clusters (including noise points labeled -1)."
    )

    # Save the DBSCAN cluster labels.
    # These labels associate each transaction (row in the original data) with a cluster ID.
    # Noise points identified by DBSCAN are labeled as -1.
    labels_output_path = os.path.join(model_dir, "cluster_labels.npy")
    np.save(labels_output_path, cluster_labels)
    print(
        f"Saved DBSCAN cluster labels ({len(np.unique(cluster_labels))} unique clusters including noise) to {labels_output_path}"
    )
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # --- Graph Construction ---
    # Nodes in the graph represent the non-noise clusters identified by DBSCAN.
    # Edges are created between clusters if their mean transaction 'Time' is similar (within a 12-hour window).
    # This aims to connect clusters that are temporally related, based on the original script logic.
    clusters = df["cluster"].unique()  # Get unique cluster IDs from DataFrame, including -1 for noise.
    # Filter out the noise cluster (-1) for graph construction.
    clusters_for_graph = [c for c in clusters if c != -1]
    # Map original cluster IDs to 0-indexed graph node IDs for Networkit.
    cluster_nodes = {c: i for i, c in enumerate(clusters_for_graph)}
    num_graph_nodes = len(cluster_nodes)

    if num_graph_nodes == 0:
        print(
            "No non-noise clusters found by DBSCAN. Cannot build graph or perform community detection."
        )
        # Save minimal artifacts and exit if no clusters are formed.
        community_assignments_path = os.path.join(
            model_dir, "community_assignments.csv"
        )
        with open(community_assignments_path, "w") as f:
            f.write(
                "graph_node_id,original_cluster_id,community_id,is_fraudulent_cluster\n"
            )
        print(f"Saved empty community assignments to {community_assignments_path}")
        print("SageMaker training script finished (no clusters to process).")
        return  # Exit early

    # Create an unweighted, undirected graph, as per original logic.
    print(f"Constructing graph with {num_graph_nodes} nodes (non-noise clusters)...")
    g = nk.graph.Graph(num_graph_nodes, weighted=False, directed=False)
    
    # Calculate the mean 'Time' for each valid (non-noise) cluster.
    # This uses the 'Time' column from the original DataFrame `df`.
    mean_times = df[df["cluster"] != -1].groupby("cluster")["Time"].mean()

    print("Adding edges based on mean transaction time similarity (within 12 hours)...")
    edge_count = 0
    # Iterate over unique pairs of clusters to determine edges.
    for i in range(len(clusters_for_graph)):
        for j in range(i + 1, len(clusters_for_graph)):
            c1 = clusters_for_graph[i]
            c2 = clusters_for_graph[j]
            
            # Check if both clusters exist in mean_times (they should, as -1 was filtered).
            if c1 in mean_times and c2 in mean_times:
                # Connect clusters if their mean transaction times are within a 12-hour window (43200 seconds).
                # This was the original logic for establishing relationships between clusters.
                if abs(mean_times[c1] - mean_times[c2]) < (3600 * 12):
                    g.addEdge(cluster_nodes[c1], cluster_nodes[c2])
                    edge_count += 1
    print(
        f"Graph constructed: {g.numberOfNodes()} nodes, {g.numberOfEdges()} edges (verified edge additions: {edge_count})."
    )

    # --- Community Detection (Louvain method using PLM) ---
    # Detect communities within the graph of clusters. Each community will group similar clusters.
    # PLM (Parallel Louvain Method) is an efficient algorithm for community detection in large networks.
    print(
        f"Graph constructed with {g.numberOfNodes()} nodes and {g.numberOfEdges()} edges."
    )
    communities = []  # List to store community ID for each graph node.
    # Handle cases where the graph has no edges (e.g., all clusters are too dissimilar).
    if g.numberOfEdges() == 0:
        print(
            "Graph has no edges. Community detection will result in trivial communities (each node its own community)."
        )
        # If no edges, each node (cluster) is considered its own community.
        # The community ID can be the node ID itself or a unique ID for each.
        communities = [
            i for i in range(g.numberOfNodes())
        ]  # Each node forms its own community.
        print(
            f"Assigned trivial communities: {len(np.unique(communities))} communities for {g.numberOfNodes()} nodes."
        )
    else:
        print("Running Louvain (PLM) community detection with default parameters...")
        # PLM (Parallel Louvain Method) is used for community detection.
        # Reverting to the simpler, likely original call without explicit refine/par arguments.
        plm = nk.community.PLM(g) # Original call with default parameters
        plm.run() # Execute the algorithm.
        communities_partition = plm.getPartition() # Get the resulting partition.

        if communities_partition.numberOfSubsets() > 0:
            # `getVector()`: Returns a list where the i-th element is the community ID of node i.
            communities = communities_partition.getVector()
            # Calculate modularity using the Modularity class and the getQuality method.
            mod_score = nk.community.Modularity().getQuality(communities_partition, g)
            print(f"Detected {communities_partition.numberOfSubsets()} communities (Louvain modularity: {mod_score:.4f}).")
        else:
            # Fallback if PLM doesn't detect communities or returns an empty partition.
            print("Louvain (PLM) did not detect any communities or returned an empty partition. Assigning all nodes to a single community (0).")
            communities = [0] * g.numberOfNodes()  # Assign all nodes to community 0.

    # --- Fraud Labeling for Clusters ---
    # Determine if a cluster (identified by DBSCAN) is predominantly fraudulent.
    # A cluster is marked as fraudulent if more than 50% of its transactions are fraudulent.
    # This creates a 'fraud label' for each cluster, not for individual transactions.
    fraud_labels = df[
        "Class"
    ].values  # Get the ground truth fraud labels (0 for non-fraud, 1 for fraud) from the 'Class' column.
    unique_clusters = np.unique(
        cluster_labels
    )  # Get all unique cluster IDs, including -1 for noise.
    fraud_labels_dict = (
        {}
    )  # Dictionary to store the fraud status of each cluster (1 if fraudulent, 0 otherwise).
    if -1 in df["cluster"].unique():  # Check if noise cluster exists
        fraud_labels_series = (
            df[df["cluster"] != -1]
            .groupby("cluster")["Class"]
            .agg(lambda x: int(x.mean() > 0.5))
        )
        fraud_labels_dict = fraud_labels_series.to_dict()
    else:
        fraud_labels_series = df.groupby("cluster")["Class"].agg(
            lambda x: int(x.mean() > 0.5)
        )
        fraud_labels_dict = fraud_labels_series.to_dict()
    print("Assigned fraud labels to clusters based on majority vote.")

    # --- Community Fraud Analysis ---
    # Analyze the concentration of fraudulent clusters within each detected community.
    # This helps in identifying high-risk communities.
    print("Analyzing fraud concentration within communities...")
    comm_fraud_stats = {}  # Dictionary to store fraud statistics per community.
    # `communities` is a list where `communities[i]` is the community ID of graph node `i`.
    # `clusters_for_graph[i]` is the original DBSCAN cluster ID for graph node `i`.
    for graph_node_idx, community_id in enumerate(communities):
        # Get the original DBSCAN cluster ID corresponding to the current graph node index.
        original_cluster_id = clusters_for_graph[graph_node_idx]
        # Get the fraud status (0 or 1) of this original cluster from the previously computed `fraud_labels_dict`.
        is_fraudulent_cluster = fraud_labels_dict.get(
            original_cluster_id, 0
        )  # Default to 0 if somehow not found.

        # Initialize stats for a new community if encountered for the first time.
        if community_id not in comm_fraud_stats:
            comm_fraud_stats[community_id] = {
                "total_clusters": 0,
                "fraudulent_clusters": 0,
            }

        # Increment counts for the community this cluster belongs to.
        comm_fraud_stats[community_id]["total_clusters"] += 1
        comm_fraud_stats[community_id]["fraudulent_clusters"] += is_fraudulent_cluster

    print(
        "\nTop 10 communities by fraud ratio (fraudulent clusters / total clusters in community):"
    )
    # Sort communities by the ratio of fraudulent clusters.
    top_communities = sorted(
        comm_fraud_stats.items(),
        key=lambda item: (
            item[1]["fraudulent_clusters"] / item[1]["total_clusters"]
            if item[1]["total_clusters"] > 0
            else 0
        ),
        reverse=True,
    )[:10]

    for comm_id, stats in top_communities:
        ratio = (
            stats["fraudulent_clusters"] / stats["total_clusters"]
            if stats["total_clusters"] > 0
            else 0
        )
        print(
            f"Community {comm_id}: {stats['fraudulent_clusters']}/{stats['total_clusters']} fraudulent clusters ({ratio:.2%})"
        )

    # --- Save Results ---
    # Save the community assignments for each graph node (which represents an original DBSCAN cluster).
    # This file maps each original cluster (via its graph node ID) to a community ID and its fraud status.
    # This is a key artifact for the `serve.py` script.
    community_assignments_path = os.path.join(model_dir, "community_assignments.csv")
    print(f"Saving community assignments to {community_assignments_path}...")
    with open(community_assignments_path, "w") as f:
        f.write(
            "graph_node_id,original_cluster_id,community_id,is_fraudulent_cluster\n"
        )
        for graph_node_idx, community_id in enumerate(communities):
            original_cluster_id = clusters_for_graph[graph_node_idx]
            is_fraudulent_cluster = fraud_labels_dict.get(original_cluster_id, 0)
            f.write(
                f"{graph_node_idx},{original_cluster_id},{community_id},{is_fraudulent_cluster}\n"
            )
    print(f"Saved community assignments to {community_assignments_path}")
    print("SageMaker training script finished successfully.")


# --- Main Execution Guard ---
# Ensures that main() is called only when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    main()
