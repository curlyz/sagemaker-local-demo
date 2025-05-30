import sagemaker
from sagemaker.pytorch import PyTorch
import os


def main():
    sagemaker_session = sagemaker.LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    try:
        role = sagemaker.get_execution_role()
        print(f"Using SageMaker execution role from AWS config: {role}")
    except (
        Exception
    ):  # Catches ValueError or any other exception if role cannot be obtained
        print(
            "SageMaker execution role not found or error obtaining it. Using a dummy role for local mode."
        )
        role = "arn:aws:iam::111111111111:role/DummyRoleForLocal"

    source_dir = os.path.abspath(os.path.dirname(__file__))
    entry_point = "train.py"
    requirements_file = "requirements.txt"

    pytorch_version = "2.2.0"
    python_version = "py310"

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
    # We'll try without it first.

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        framework_version=pytorch_version,
        py_version=python_version,
        instance_count=1,
        instance_type="local",  # Crucial for local mode
        sagemaker_session=sagemaker_session,
        requirements_file=requirements_file,
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker",  # Standard x86_64 training image for testing
        hyperparameters={"epochs": 1},
        model_dir="./model"  # Ensures /opt/ml/model is mapped to ./model locally
    )

    print(
        "Starting local training job. This might take a while for the first run as Docker images are pulled and built..."
    )
    try:
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


if __name__ == "__main__":
    main()
