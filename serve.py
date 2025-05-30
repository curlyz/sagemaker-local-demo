"""
This script defines a FastAPI application for serving fraud detection predictions.

It loads pre-computed model artifacts (DBSCAN cluster labels, a StandardScaler,
community assignments from Networkit, and feature data) to make predictions.
The API provides:
- A `/health` endpoint to check if the service is ready and artifacts are loaded.
- A `/predict` endpoint that takes transaction features (V1-V28) and returns
  the assigned community, the fraud ratio of that community, and the assigned node ID.

The script handles artifact loading at startup and ensures they are available for prediction.
It also includes a global exception handler for better error reporting.
It can be run locally using Uvicorn or deployed as a SageMaker endpoint.
"""

import uvicorn  # ASGI server for running FastAPI applications
from fastapi import FastAPI, Request  # FastAPI framework components
import pandas as pd  # For data manipulation, especially reading CSVs
import numpy as np  # For numerical operations, especially array handling
import networkit as nk  # type: ignore # For graph analytics and community detection (used in training, artifacts loaded here)
import pickle  # For loading a pickled scikit-learn scaler object
import os  # For interacting with the operating system, e.g., path manipulation, environment variables
from sklearn.cluster import DBSCAN  # DBSCAN clustering algorithm (used in training)
from sklearn.preprocessing import (
    StandardScaler,
)  # For feature scaling (used in training and prediction)
from typing import Tuple, Dict, Any  # For type hinting

import traceback  # For formatting exception tracebacks
from fastapi.responses import (
    JSONResponse,
)  # For returning JSON responses, especially for errors
from fastapi import (
    status,
)  # HTTP status codes (FastAPI also re-exports `Request` here, but it's already imported)

# Initialize the FastAPI application instance
app = FastAPI()


# Global exception handler for the FastAPI application.
# This catches any unhandled exceptions during request processing.
@app.exception_handler(Exception)
async def traceback_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions and returns a JSON response
    with the error message and a traceback for debugging.
    Args:
        request (Request): The incoming request that caused the exception.
        exc (Exception): The exception that was raised.
    Returns:
        JSONResponse: A 500 error response with error details.
    """
    tb = traceback.format_exc()  # Get the formatted traceback string.
    # Return a JSON response with HTTP status 500 (Internal Server Error).
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  # Use status constant
        content={
            "error": str(exc),
            "traceback": tb,
        },  # Include error message and traceback in the response body.
    )


# --- Define Paths for Model Artifacts ---
# The script needs to locate model artifacts (scaler, cluster labels, etc.).
# It prioritizes a local './model' directory for development convenience.
# If './model' doesn't exist, it falls back to the SageMaker environment variable `SM_MODEL_DIR`,
# which is standard for SageMaker hosting. If neither is found, it defaults to the current directory (".").

# Determine the base directory for model artifacts.
if os.path.exists("./model"):  # Check for a local './model' directory first.
    MODEL_DIR = "./model"
    print(f"Using local model directory: {MODEL_DIR}")
elif os.environ.get("SM_MODEL_DIR"):
    MODEL_DIR = os.environ.get(
        "SM_MODEL_DIR"
    )  # Use SageMaker's standard model directory if available.
    print(f"Using SageMaker model directory: {MODEL_DIR}")
else:
    MODEL_DIR = "."  # Default to current directory if others are not found.
    print(
        f"Warning: Neither './model' nor SM_MODEL_DIR found. Using current directory for models: {MODEL_DIR}"
    )
# Define full paths to individual artifact files within the MODEL_DIR.
CLUSTER_PATH = os.path.join(
    MODEL_DIR, "cluster_labels.npy"
)  # Path to DBSCAN cluster labels.
SCALER_PATH = os.path.join(
    MODEL_DIR, "scaler.pkl"
)  # Path to the pickled StandardScaler object.
COMMUNITY_PATH = os.path.join(
    MODEL_DIR, "community_assignments.csv"
)  # Path to community assignments from Networkit.
FEATURES_PATH = os.path.join(
    MODEL_DIR, "creditcard_features.csv"
)  # Path to the features used for calculating cluster centers.


# --- Artifact Loading Function ---
# This function is responsible for loading the core artifacts needed for prediction.


def load_artifacts() -> Tuple[np.ndarray, StandardScaler, Dict[int, Dict[str, Any]]]:
    """Load cluster labels, scaler, and community assignments from disk.

    Returns:
        Tuple containing:
            - cluster_labels (np.ndarray): Array of cluster labels from DBSCAN.
            - scaler (StandardScaler): Fitted StandardScaler object.
            - comm_map (Dict[int, Dict[str, Any]]): Dictionary mapping node IDs to their
                                                    community and fraud status.
    Raises:
        FileNotFoundError: If any of the required artifact files are not found.
    """
    print(f"Loading cluster labels from: {CLUSTER_PATH}")
    cluster_labels: np.ndarray = np.load(CLUSTER_PATH)

    print(f"Loading scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    print(f"Loading community assignments from: {COMMUNITY_PATH}")
    comm_df: pd.DataFrame = pd.read_csv(COMMUNITY_PATH)
    # Convert the community DataFrame into a dictionary for easier lookup.
    # The dictionary maps node_id to a sub-dictionary containing 'community' and 'is_fraud'.
    comm_map: Dict[int, Dict[str, Any]] = comm_df.set_index("node_id")[
        ["community", "is_fraud"]
    ].to_dict("index")

    print("Successfully loaded cluster_labels, scaler, and comm_map.")
    return cluster_labels, scaler, comm_map


# --- Global Variables for Loaded Artifacts ---
# These variables will hold the loaded model artifacts once `ensure_artifacts_loaded` is called.
# They are initialized to None and populated at startup or on first request to /health or /predict.
cluster_labels: np.ndarray | None = None
scaler: StandardScaler | None = None
comm_map: Dict[int, Dict[str, Any]] | None = None
cluster_centers: Dict[int, np.ndarray] | None = (
    None  # Dictionary to store mean feature vectors for each cluster
)

from fastapi.responses import JSONResponse
from fastapi import status


# --- Ensure Artifacts are Loaded (Idempotent) ---
# This function checks if artifacts are already loaded and, if not, loads them.
# It also calculates cluster centers based on the loaded features and cluster labels.


def ensure_artifacts_loaded():
    """
    Ensures that all necessary model artifacts are loaded into global variables.
    This function is idempotent; it will only load artifacts if they haven't been loaded yet.

    It performs the following steps:
    1. Checks if global artifact variables (cluster_labels, scaler, comm_map, cluster_centers) are already populated.
       If so, it returns early.
    2. Verifies the existence of all required artifact files on disk.
       Raises FileNotFoundError if any are missing.
    3. Calls `load_artifacts()` to load cluster labels, scaler, and community map.
    4. Loads the raw feature data from `FEATURES_PATH`.
    5. Calculates the mean feature vector for each cluster (cluster centers), excluding noise points (label -1).
    6. Assigns the loaded artifacts and calculated cluster centers to the global variables.
    """
    global cluster_labels, scaler, comm_map, cluster_centers
    # Check if artifacts are already loaded to prevent redundant loading.
    # This makes the function idempotent.
    if (
        cluster_labels is not None
        and scaler is not None
        and comm_map is not None
        and cluster_centers is not None
    ):
        print("Artifacts already loaded. Skipping.")
        return
    print("Attempting to load artifacts...")
    # Verify that all required artifact files exist before attempting to load.
    if not (
        os.path.exists(CLUSTER_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(COMMUNITY_PATH)
        and os.path.exists(FEATURES_PATH)
    ):
        raise FileNotFoundError(
            f"One or more model artifacts are missing in '{MODEL_DIR}'. Required files: 'cluster_labels.npy', 'scaler.pkl', 'community_assignments.csv', 'creditcard_features.csv'. Please ensure they are present (e.g., copied from training output)."
        )

    # Load the primary artifacts.
    cluster_labels_local, scaler_local, comm_map_local = load_artifacts()
    # Load the raw feature data, which is used to calculate cluster centers.
    print(f"Loading features for cluster center calculation from: {FEATURES_PATH}")
    features_all: np.ndarray = np.genfromtxt(
        FEATURES_PATH, delimiter=",", skip_header=1  # Assumes CSV with a header row.
    )
    # The original dataset might have an ID or index column as the first column.
    # If the shape indicates 29 columns, assume the first is an index and drop it to get the 28 features (V1-V28).
    # This ensures consistency with the features expected by the prediction endpoint.
    if features_all.shape[1] == 29:
        print(
            f"Original features_all shape: {features_all.shape}. Dropping first column (assumed index)."
        )
        features_all = features_all[
            :, 1:
        ]  # Select all rows, and columns from the second one onwards.
        print(f"New features_all shape: {features_all.shape}.")
    # Calculate cluster centers (mean feature vector for each cluster).
    # These centers are used in the /predict endpoint to assign new data points to the nearest cluster.
    unique_clusters = np.unique(cluster_labels_local)  # Get all unique cluster labels.
    print(f"Found unique cluster labels: {unique_clusters}")
    cluster_centers_local: Dict[int, np.ndarray] = {}
    for c_label in unique_clusters:
        if (
            c_label == -1
        ):  # DBSCAN uses -1 for noise points; do not calculate a center for noise.
            print(f"Skipping cluster center calculation for noise points (label -1).")
            continue
        # Select features belonging to the current cluster label 'c_label'.
        features_in_cluster = features_all[cluster_labels_local == c_label]
        if features_in_cluster.size == 0:
            print(
                f"Warning: Cluster {c_label} has no features assigned. Skipping center calculation for this cluster."
            )
            continue
        # Calculate the mean of these features to get the cluster center.
        cluster_centers_local[int(c_label)] = features_in_cluster.mean(axis=0)
        print(f"Calculated center for cluster {c_label}.")
    # Assign the loaded and calculated artifacts to the global variables.
    cluster_labels = cluster_labels_local
    scaler = scaler_local
    comm_map = comm_map_local
    cluster_centers = cluster_centers_local
    print("All artifacts loaded and cluster centers calculated successfully.")


ensure_artifacts_loaded()


# --- FastAPI Event Handlers ---


# This event handler is triggered when the FastAPI application starts up.
@app.on_event("startup")
def startup_event():
    """
    Performs application startup tasks, primarily ensuring model artifacts are loaded.
    If artifacts are missing, it prints an error message but allows the server to start
    (the /health and /predict endpoints will then report issues).
    """
    print("FastAPI application startup event triggered.")
    try:
        ensure_artifacts_loaded()  # Attempt to load all necessary model artifacts.
        print("Startup: Artifacts loaded (or were already loaded).")
    except FileNotFoundError as e:
        # Log the error if artifacts are not found during startup.
        # The server will still start, but /health and /predict will be affected.
        print(f"Startup Error: Could not load artifacts. {e}")


# --- API Endpoints ---


# Health check endpoint.
@app.get("/health")
def health():
    """
    Provides a health check for the service.
    It attempts to ensure artifacts are loaded. If successful, returns an 'ok' status.
    If artifacts are missing, it returns a 503 Service Unavailable status.
    Returns:
        Dict or JSONResponse: Status of the service.
    """
    print("Health check endpoint called.")
    try:
        ensure_artifacts_loaded()  # Ensure artifacts are ready.
        print("Health check: Artifacts loaded successfully.")
        return {
            "status": "ok",
            "message": "Service is healthy and artifacts are loaded.",
        }
    except FileNotFoundError as e:
        print(f"Health check: Failed - Artifacts not found. {e}")
        # Return a 503 Service Unavailable if critical artifacts are missing.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Service is unavailable due to missing model artifacts.",
                "detail": str(e),
            },
        )


# Prediction endpoint.
@app.post("/predict")
async def predict(request: Request) -> Dict[str, Any]:
    """
    Receives transaction features (V1-V28) and predicts the community,
    community fraud ratio, and the assigned node ID.

    The prediction logic involves:
    1. Ensuring model artifacts are loaded.
    2. Scaling the input features using the pre-fitted scaler.
    3. Assigning the scaled features to the nearest pre-calculated cluster center.
       The ID of this cluster is treated as the 'node_id'.
    4. Looking up the community and fraud status for this 'node_id' from `comm_map`.
    5. Calculating the overall fraud ratio for the assigned community.

    Args:
        request (Request): The FastAPI request object, expected to contain a JSON body
                           with keys 'V1' through 'V28'.

    Returns:
        Dict[str, Any]: A dictionary containing 'community', 'community_fraud_ratio',
                        and 'node_id'.

    Raises:
        JSONResponse: If artifacts are not loaded (via `ensure_artifacts_loaded` indirectly
                      raising FileNotFoundError, caught by the global exception handler or health check first).
                      Or if input data is malformed (would also be caught by global handler).
    """
    print("Predict endpoint called.")
    ensure_artifacts_loaded()  # Crucial: make sure all models/data are loaded before attempting prediction.
    body = await request.json()  # Get the JSON payload from the request.

    # Extract V1-V28 features from the request body and convert to a NumPy array.
    # Assumes the input JSON has keys like "V1", "V2", ..., "V28".
    try:
        features = np.array([body[f"V{i}"] for i in range(1, 29)]).reshape(
            1, -1
        )  # Reshape to (1, 28) for scaler
    except KeyError as e:
        print(f"Prediction error: Missing feature in input data - {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Missing feature in input data: {e}. Expected V1-V28."},
        )

    print(f"Received features for prediction: {features.shape}")
    # Scale the input features using the loaded StandardScaler.
    # The scaler must not be None here due to ensure_artifacts_loaded().
    features_scaled = scaler.transform(features)
    print(f"Scaled features: {features_scaled.shape}")
    # Assign the new data point to the nearest cluster center.
    # The `cluster_centers` dictionary (mapping cluster_id to its mean feature vector)
    # must be populated by `ensure_artifacts_loaded()`.
    min_dist = float("inf")
    assigned_cluster_id = -1  # Default to -1 (like noise or unassigned)

    if not cluster_centers:  # Should not happen if ensure_artifacts_loaded worked
        print("Prediction error: Cluster centers not loaded.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Cluster centers not available."},
        )

    for c_id, center_vec in cluster_centers.items():
        dist = np.linalg.norm(
            features_scaled - center_vec
        )  # Calculate Euclidean distance.
        if dist < min_dist:
            min_dist = dist
            assigned_cluster_id = c_id

    # The assigned_cluster_id is treated as the 'node_id' for looking up community info.
    node_id = assigned_cluster_id
    print(
        f"Assigned input to node_id (cluster_id): {node_id} with distance: {min_dist}"
    )
    # Retrieve community information for the assigned node_id.
    # `comm_map` (mapping node_id to community and fraud status) must be loaded.
    # Default to a generic 'unknown' community if node_id is not found (e.g., if assigned_cluster_id was -1 or not in comm_map).
    community_info = comm_map.get(
        node_id, {"community": -1, "is_fraud": 0}
    )  # is_fraud from this specific node is not directly used here.
    assigned_community_id = community_info["community"]
    print(f"Node {node_id} belongs to community {assigned_community_id}.")

    # Calculate the fraud ratio for the entire assigned community.
    # This involves iterating through all nodes in `comm_map` that belong to `assigned_community_id`.
    if assigned_community_id != -1 and comm_map:
        frauds_in_community = [
            v["is_fraud"]
            for v in comm_map.values()
            if v["community"] == assigned_community_id
        ]
        community_fraud_ratio = (
            sum(frauds_in_community) / len(frauds_in_community)
            if frauds_in_community
            else 0.0
        )
    else:
        community_fraud_ratio = 0.0  # Or handle as 'unknown'/'N/A'
        print(
            f"Could not calculate fraud ratio for community {assigned_community_id} (community is -1 or comm_map empty)."
        )
    print(
        f"Calculated fraud ratio for community {assigned_community_id}: {community_fraud_ratio}"
    )
    return {
        "community": int(assigned_community_id),
        "community_fraud_ratio": community_fraud_ratio,
        "node_id": int(node_id),  # This is the cluster ID the input was assigned to.
    }


# --- Main Execution Block ---
# This block allows the script to be run directly using Uvicorn for local development.
if __name__ == "__main__":
    print("Starting Uvicorn server for local development...")
    # `reload=True` enables auto-reloading when code changes, useful for development.
    # `host="0.0.0.0"` makes the server accessible from other devices on the network.
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
