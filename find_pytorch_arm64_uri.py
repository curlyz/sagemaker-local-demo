"""
This script attempts to find a SageMaker Docker image URI for PyTorch
running on ARM64 (Graviton) instances.

It uses the SageMaker SDK's `image_uris.retrieve()` method to find suitable
Docker image URIs based on specified framework, region, version, Python version,
instance type, and image scope (training).

The script makes two attempts with different PyTorch versions if the first one fails:
1. PyTorch 2.2.1 with Python 3.10
2. PyTorch 2.1.0 with Python 3.10 (as a fallback)

This is particularly useful when working with newer ARM-based SageMaker instances
where image availability might differ from x86_64 instances.
"""
from sagemaker import image_uris  # type: ignore

# --- Configuration for SageMaker Image URI retrieval ---
framework = "pytorch"  # The machine learning framework
region = "us-west-2"      # AWS region, should match your SageMaker execution environment
instance_type = "ml.m6g.large"  # Specifies an ARM-based (Graviton) instance type
image_scope = "training"    # Specifies that the image should be for training (not inference)

# --- Attempt 1: Using PyTorch 2.2.1 and Python 3.10 ---
# These versions are typically listed as supported by the SageMaker SDK.
framework_version_1 = "2.2.1"  # Target PyTorch framework version
py_version_1 = "py310"        # Target Python version (e.g., py310 for Python 3.10)

print(
    f"Attempt 1: Trying to find SageMaker URI for PyTorch {framework_version_1}, Python {py_version_1}, ARM64, scope: {image_scope}..."
)
try:
    # Attempt to retrieve the Docker image URI using the specified parameters.
    uri_1 = image_uris.retrieve(
        framework=framework,
        region=region,
        version=framework_version_1,
        py_version=py_version_1,
        instance_type=instance_type,
        image_scope=image_scope,
    )
    # Verify if the found URI is indeed for training, as image_scope can sometimes be ignored or lead to ambiguous results.
    if "training" in uri_1:
        print(f"Success (Attempt 1): Found TRAINING URI: {uri_1}")
    else:
        print(
            f"Warning (Attempt 1): Found URI, but it might not be a training image (please verify): {uri_1}"
        )
        # Explicitly check if an inference image was returned by mistake.
        if "inference" in uri_1:
            raise ValueError(f"Error (Attempt 1): Found an inference image instead of a training image: {uri_1}")

except Exception as e1:
    # This block catches any errors during the first attempt (e.g., image not found).
    print(
        f"Failed (Attempt 1): Could not find a suitable TRAINING URI for PyTorch {framework_version_1}, Python {py_version_1}. Error: {e1}"
    )

    # --- Attempt 2: Using PyTorch 2.1.0 and Python 3.10 (Fallback) ---
    # If the first attempt fails, try a slightly older, potentially more stable/available version.
    framework_version_2 = "2.1.0"  # Fallback PyTorch framework version
    py_version_2 = "py310"        # Fallback Python version
    print(
        f"\nAttempt 2: Trying to find SageMaker URI for PyTorch {framework_version_2}, Python {py_version_2}, ARM64, scope: {image_scope}..."
    )
    try:
        # Attempt to retrieve the Docker image URI using the fallback parameters.
        uri_2 = image_uris.retrieve(
            framework=framework,
            region=region,
            version=framework_version_2,
            py_version=py_version_2,
            instance_type=instance_type,
            image_scope=image_scope,
        )
        # Verify if the found URI is indeed for training.
        if "training" in uri_2:
            print(f"Success (Attempt 2): Found TRAINING URI: {uri_2}")
        else:
            print(
                f"Warning (Attempt 2): Found URI, but it might not be a training image (please verify): {uri_2}"
            )
            # Explicitly check if an inference image was returned by mistake.
            if "inference" in uri_2:
                raise ValueError(f"Error (Attempt 2): Found an inference image instead of a training image: {uri_2}")

    except Exception as e2:
        # This block catches any errors during the second attempt.
        print(
            f"Failed (Attempt 2): Could not find a suitable TRAINING URI for PyTorch {framework_version_2}, Python {py_version_2}. Error: {e2}"
        )
        print(
            "\nConclusion: Failed to find a suitable ARM64 PyTorch training image with the tried SDK-supported versions."
        )
        # Consider manually checking the AWS documentation for available SageMaker images for ARM64 instances
        # or trying other framework/Python version combinations.
