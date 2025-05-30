import argparse
import os
import sys

n_cores = os.cpu_count() or 1
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

torch.set_num_threads(n_cores)
nk.setNumberOfThreads(n_cores)


def main():
    model_dir = os.environ.get("SM_MODEL_DIR", ".")
    print("Starting SageMaker training script.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Networkit version: {nk.__version__}")
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load Kaggle credit card fraud dataset
    data_path = "creditcard.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset {data_path} not found. Please download from Kaggle and place in the working directory."
        )
    if model_dir == ".":
        df = pd.read_csv(data_path, nrows=5000)  # Limit to 5000 rows for local dev
    else:
        df = pd.read_csv(data_path)
    print(f"Loaded Kaggle creditcard.csv: {df.shape[0]} transactions")

    # Refined: Cluster transactions (V1-V28) using DBSCAN, treat clusters as nodes
    features = df[[f"V{i}" for i in range(1, 29)]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Save scaled features with index for mapping
    features_df = pd.DataFrame(features_scaled, columns=[f"V{i}" for i in range(1, 29)])
    features_df.insert(0, "index", df.index)
    features_df.to_csv(os.path.join(model_dir, "creditcard_features.csv"), index=False)
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    cluster_labels = dbscan.fit_predict(features_scaled)
    df["cluster"] = cluster_labels
    np.save(os.path.join(model_dir, "cluster_labels.npy"), cluster_labels)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Build graph: nodes = clusters, edges between clusters with similar mean time
    clusters = df["cluster"].unique()
    cluster_nodes = {c: i for i, c in enumerate(clusters)}
    g = nk.graph.Graph(len(cluster_nodes), weighted=False, directed=False)
    mean_times = df.groupby("cluster")["Time"].mean()
    for c1 in clusters:
        for c2 in clusters:
            if (
                c1 < c2 and abs(mean_times[c1] - mean_times[c2]) < 3600 * 12
            ):  # 12 hours proximity
                g.addEdge(cluster_nodes[c1], cluster_nodes[c2])
    print(f"Graph: {g.numberOfNodes()} nodes, {g.numberOfEdges()} edges")

    # Assign fraud labels to clusters (majority vote)
    fraud_labels = (
        df.groupby("cluster")["Class"].agg(lambda x: int(x.mean() > 0.5)).to_dict()
    )

    # Louvain community detection (PLM is Louvain in Networkit)
    plm = nk.community.PLM(g)
    plm.run()
    communities = plm.getPartition().getVector()
    print(f"Detected {max(communities)+1} communities")

    # Analyze fraud concentration by community
    comm_fraud = {}
    for node, comm in enumerate(communities):
        is_fraud = fraud_labels.get(node, 0)
        if comm not in comm_fraud:
            comm_fraud[comm] = {"total": 0, "fraud": 0}
        comm_fraud[comm]["total"] += 1
        comm_fraud[comm]["fraud"] += is_fraud
    print("\nTop 10 communities by fraud ratio:")
    top = sorted(
        comm_fraud.items(),
        key=lambda x: (x[1]["fraud"] / x[1]["total"] if x[1]["total"] > 0 else 0),
        reverse=True,
    )[:10]
    for comm, stats in top:
        print(
            f"Community {comm}: {stats['fraud']}/{stats['total']} fraud ({stats['fraud']/stats['total']:.2%})"
        )

    # Save community assignments
    out_path = os.path.join(model_dir, "community_assignments.csv")
    with open(out_path, "w") as f:
        f.write("node_id,community,is_fraud\n")
        for node, comm in enumerate(communities):
            is_fraud = fraud_labels.get(node, 0)
            f.write(f"{node},{comm},{is_fraud}\n")
    print(f"Saved community assignments to {out_path}")
    print("SageMaker training script finished successfully.")


if __name__ == "__main__":
    main()
