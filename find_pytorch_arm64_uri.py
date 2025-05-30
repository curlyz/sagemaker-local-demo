from sagemaker import image_uris  # type: ignore

framework = "pytorch"
region = "us-west-2"  # Matching your profile's region
instance_type = "ml.m6g.large"  # ARM instance
image_scope = "training"  # Explicitly training

# Attempt 1: PyTorch 2.2.1 with Python 3.10 (from SDK supported list)
framework_version_1 = "2.2.1"
py_version_1 = "py310"

print(
    f"Attempting to find URI for PT {framework_version_1}, {py_version_1}, ARM64, scope: {image_scope}..."
)
try:
    uri_1 = image_uris.retrieve(
        framework=framework,
        region=region,
        version=framework_version_1,
        py_version=py_version_1,
        instance_type=instance_type,
        image_scope=image_scope,
    )
    if "training" in uri_1:
        print(f"Found TRAINING URI: {uri_1}")
    else:
        print(
            f"Found URI (but it might not be a training image, please verify): {uri_1}"
        )
        if "inference" in uri_1:
            raise ValueError(f"Found inference image, not training: {uri_1}")

except Exception as e1:
    print(
        f"Could not find a suitable TRAINING URI for PT {framework_version_1}, {py_version_1}: {e1}"
    )

    # Attempt 2: PyTorch 2.1.0 with Python 3.10 (fallback from SDK supported list)
    framework_version_2 = "2.1.0"
    py_version_2 = "py310"
    print(
        f"\nAttempting to find URI for PT {framework_version_2}, {py_version_2}, ARM64, scope: {image_scope}..."
    )
    try:
        uri_2 = image_uris.retrieve(
            framework=framework,
            region=region,
            version=framework_version_2,
            py_version=py_version_2,
            instance_type=instance_type,
            image_scope=image_scope,
        )
        if "training" in uri_2:
            print(f"Found TRAINING URI: {uri_2}")
        else:
            print(
                f"Found URI (but it might not be a training image, please verify): {uri_2}"
            )
            if "inference" in uri_2:
                raise ValueError(f"Found inference image, not training: {uri_2}")

    except Exception as e2:
        print(
            f"Could not find a suitable TRAINING URI for PT {framework_version_2}, {py_version_2}: {e2}"
        )
        print(
            "\nFailed to find a suitable ARM64 PyTorch training image with the tried versions supported by the SDK."
        )
