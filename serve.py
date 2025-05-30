import uvicorn
from fastapi import FastAPI, Request
import pandas as pd
import numpy as np
import networkit as nk  # type: ignore
import pickle
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

import traceback
from fastapi.responses import JSONResponse
from fastapi import status, Request

app = FastAPI()


@app.exception_handler(Exception)
async def traceback_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": tb},
    )


# Always use ./model for local dev if it exists
if os.path.exists("./model"):
    MODEL_DIR = "./model"
else:
    MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
CLUSTER_PATH = os.path.join(MODEL_DIR, "cluster_labels.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COMMUNITY_PATH = os.path.join(MODEL_DIR, "community_assignments.csv")
FEATURES_PATH = os.path.join(MODEL_DIR, "creditcard_features.csv")


# Load cluster labels, scaler, and community assignments
def load_artifacts() -> (
    Tuple[np.ndarray, StandardScaler, Dict[int, Dict[str, Any]], np.ndarray]
):
    """Load cluster labels, scaler, community assignments, and features."""
    cluster_labels: np.ndarray = np.load(CLUSTER_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    comm_df: pd.DataFrame = pd.read_csv(COMMUNITY_PATH)
    comm_map: Dict[int, Dict[str, Any]] = comm_df.set_index("node_id")[
        ["community", "is_fraud"]
    ].to_dict("index")
    return cluster_labels, scaler, comm_map


cluster_labels = None
scaler = None
comm_map = None
cluster_centers = None

from fastapi.responses import JSONResponse
from fastapi import status


def ensure_artifacts_loaded():
    global cluster_labels, scaler, comm_map, cluster_centers
    if (
        cluster_labels is not None
        and scaler is not None
        and comm_map is not None
        and cluster_centers is not None
    ):
        return
    # Check for existence
    if not (
        os.path.exists(CLUSTER_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(COMMUNITY_PATH)
        and os.path.exists(FEATURES_PATH)
    ):
        raise FileNotFoundError(
            f"One or more model artifacts are missing in {MODEL_DIR}. Please copy cluster_labels.npy, scaler.pkl, community_assignments.csv, and creditcard_features.csv from your training output."
        )
    cluster_labels_local, scaler_local, comm_map_local = load_artifacts()
    features_all: np.ndarray = np.genfromtxt(
        FEATURES_PATH, delimiter=",", skip_header=1
    )
    # Drop first column if it is an index or non-feature (to ensure shape matches /predict input)
    if features_all.shape[1] == 29:
        features_all = features_all[:, 1:]
    unique_clusters = np.unique(cluster_labels_local)
    cluster_centers_local: Dict[int, np.ndarray] = {
        int(c): features_all[cluster_labels_local == c].mean(axis=0)
        for c in unique_clusters
        if c != -1
    }
    cluster_labels = cluster_labels_local
    scaler = scaler_local
    comm_map = comm_map_local
    cluster_centers = cluster_centers_local


ensure_artifacts_loaded()


@app.on_event("startup")
def startup_event():
    try:
        ensure_artifacts_loaded()
    except FileNotFoundError as e:
        print(e)


@app.get("/health")
def health():
    try:
        ensure_artifacts_loaded()
        return {"status": "ok"}
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"error": str(e)}
        )


@app.post("/predict")
async def predict(request: Request) -> Dict[str, Any]:
    ensure_artifacts_loaded()
    """Predict community and fraud risk for input transaction features."""
    body = await request.json()
    # Accept flat V1-V28 keys as input
    features = np.array([body[f"V{i}"] for i in range(1, 29)]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    # Assign to nearest cluster center
    min_dist = float("inf")
    node_id = -1
    for cid, center in cluster_centers.items():
        dist = np.linalg.norm(features_scaled - center)
        if dist < min_dist:
            min_dist = dist
            node_id = cid
    comm = comm_map.get(node_id, {"community": -1, "is_fraud": 0})
    # Calculate fraud ratio for the community
    comm_id = comm["community"]
    frauds = [v["is_fraud"] for v in comm_map.values() if v["community"] == comm_id]
    fraud_ratio = sum(frauds) / len(frauds) if frauds else 0
    return {
        "community": int(comm_id),
        "community_fraud_ratio": fraud_ratio,
        "node_id": int(node_id),
    }


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
