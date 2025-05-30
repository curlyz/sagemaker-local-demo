"""
This script runs a SageMaker PyTorch training job in local mode.

It configures a SageMaker LocalSession, defines a PyTorch estimator with
specific parameters for local execution (including entry point, source directory,
framework versions, and a pre-selected Docker image URI), and then starts
the training job using `estimator.fit()`.

The script includes error handling and troubleshooting tips, particularly
relevant for users on Apple Silicon (ARM64) machines who might encounter
Docker image compatibility issues.
"""
import sagemaker
from sagemaker.pytorch import PyTorch
import os


def main():
    """
    Sets up and runs the SageMaker local training job.

    Steps involved:
    1. Initializes a SageMaker LocalSession.
    2. Attempts to get an AWS IAM execution role; uses a dummy role if not found (common for local setups).
    3. Defines paths for the source directory, entry point script (`train.py`), and requirements file.
    4. Specifies PyTorch and Python versions for the training container.
    5. Configures a PyTorch estimator with all necessary parameters, including a hardcoded
       x86_64 Docker image URI as a workaround for potential ARM64 image issues on M-series Macs.
    6. Sets hyperparameters and the local model output directory.
    7. Starts the local training job and prints success or error messages, along with troubleshooting advice.
    """
    # Initialize a SageMaker LocalSession. This session directs SageMaker SDK calls
    # to use the local environment (e.g., local Docker) instead of AWS cloud resources.
    sagemaker_session = sagemaker.LocalSession()
    # Ensure that the local code (source_dir) is used directly.
    sagemaker_session.config = {"local": {"local_code": True}}

    # Attempt to get the SageMaker execution role from the environment (e.g., AWS CLI config).
    # If not available (common in purely local setups or without AWS configured), use a dummy role.
    # The role is less critical for local mode but still required by the Estimator's constructor.
    try:
        role = sagemaker.get_execution_role()
        print(f"Successfully retrieved SageMaker execution role from AWS config: {role}")
    except Exception as e:
        print(
            f"Could not retrieve SageMaker execution role (Error: {e}). \
Using a dummy role for local mode: 'arn:aws:iam::111111111111:role/DummyRoleForLocal'"
        )
        role = "arn:aws:iam::111111111111:role/DummyRoleForLocal"  # Placeholder role

    # Define the source directory (where train.py and requirements.txt are located).
    # os.path.dirname(__file__) gets the directory of the current script (run_local_training.py).
    source_dir = os.path.abspath(os.path.dirname(__file__))
    # The main training script to be executed inside the Docker container.
    entry_point = "train.py"
    # The file listing Python package dependencies for the training script.
    requirements_file = "requirements.txt"

    # Specify the PyTorch and Python versions for the training environment.
    # These should match the versions used in your `train.py` script and `requirements.txt`.
    pytorch_version = "2.2.0"
    python_version = "py310"  # Corresponds to Python 3.10

    print(
        f"Configuring PyTorch estimator for PyTorch {pytorch_version}, Python {python_version} for local execution."
    )
    print(f"Source directory: {source_dir}")
    print(f"Entry point: {entry_point}")
    print(f"Requirements file: {requirements_file}")

    # For Apple Silicon M3, SageMaker local mode should ideally pick an ARM64 image.
    # If it fails, we might need to specify an image_uri explicitly.
    # Example ARM64 image for PyTorch 2.2, Python 3.11, CPU:
    # image_uri_arm64 = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2-cpu-py311-arm64-ubuntu22.04-sagemaker"
    # We'll try without it first, and instead use a known-good x86_64 image (see image_uri below).

    # Configure the PyTorch estimator for local training.
    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        framework_version=pytorch_version,
        py_version=python_version,
        instance_count=1,
        instance_type="local",         # Crucial: Specifies that training should run locally, not on a SageMaker instance.
        sagemaker_session=sagemaker_session, # Use the configured LocalSession.
        requirements_file=requirements_file, # Path to the requirements.txt file, relative to source_dir.
        # --- Docker Image URI --- 
        # Using a specific x86_64 image URI. This is a workaround based on previous experiences (see MEMORY)
        # where ARM64 (Graviton) images for PyTorch caused authorization errors on an M3 Mac, even with
        # successful Docker login. The standard x86_64 image runs via Rosetta 2 on M-series Macs.
        # If running on a native x86_64 Linux/Windows machine, SageMaker might auto-select an appropriate image
        # if image_uri is omitted, but explicit definition provides consistency.
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker",
        hyperparameters={"epochs": 1}, # Hyperparameters passed to the train.py script.
        # model_dir specifies where the trained model artifacts will be saved locally.
        # Inside the container, this corresponds to /opt/ml/model.
        # SageMaker local mode maps this container path to the specified local path.
        model_dir="./model",
    )

    print(
        "\nStarting local training job. This might take a while for the first run as Docker images are pulled and the environment is built..."
    )
    try:
        # This call initiates the local training process.
        # SageMaker will use Docker to run the entry_point script in the specified container environment.
        estimator.fit()
        print("\nLocal training job finished successfully!")
        print("Check the console output above for logs from 'train.py'.")
        print(
            "You should see messages indicating PyTorch and Networkit versions, and community detection results."
        )
    except Exception as e:
        print(f"\nAn error occurred during local training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Docker Desktop is running and has enough resources allocated.")
        print(
            "2. Your active Python environment (e.g., conda 'sagemaker-local') should have 'sagemaker', 'torch', 'boto3' installed."
        )
        print(
            "3. If you see errors related to Docker image architecture (e.g., 'exec format error'),"
        )
        print(
            "   it might mean an x86_64 image was pulled. Try uncommenting and using the 'image_uri_arm64'"
        )
        print(
            "   in 'run_local_training.py' with an appropriate ARM64 PyTorch image from AWS ECR."
        )
        print(
            "4. Check for errors during the 'networkit' installation phase in the Docker build logs (printed above)."
        )
        print(
            "5. Ensure the `image_uri` in this script points to a valid and accessible SageMaker training image."
        )


# Standard Python idiom: defines the script's entry point when executed directly.
if __name__ == "__main__":
    main()
