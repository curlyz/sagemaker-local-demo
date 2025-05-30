# Original source/inspiration: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-local-mode.html
"""
This script demonstrates how to set up and run a SageMaker Machine Learning Pipeline
in local mode. Local mode allows for testing pipeline definitions and logic
without provisioning AWS resources, using the local machine's Docker environment.

Key components demonstrated:
- LocalPipelineSession: For running pipelines locally.
- PyTorch Estimator: Configured for a local training job.
- TrainingStep: To encapsulate the training job within the pipeline.
- Pipeline: Definition and local execution.
- Retrieving step outputs after local execution.
"""

import sagemaker # Added import for sagemaker.get_execution_role()
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.inputs import TrainingInput # Added import for TrainingInput
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline

# --- 1. Initialize a Local Pipeline Session ---
# This session configures SageMaker SDK operations to run locally.
local_pipeline_session = LocalPipelineSession()

# --- 2. Define a SageMaker Estimator (PyTorch in this case) ---
# The estimator is configured to run with the local session.
pytorch_estimator = PyTorch(
    sagemaker_session=local_pipeline_session,  # Use the local session
    role=sagemaker.get_execution_role(),       # IAM role ARN. For local mode, this might use default local credentials or a dummy role.
                                               # Ensure your local AWS setup can resolve this.
    instance_type="local", # For local mode, 'local' or 'local_gpu' is typically used.
                               # 'ml.c5.xlarge' might be ignored or cause issues if not running on an actual SageMaker instance.
                               # Using 'local' to be explicit for local training.
    instance_count=1,
    framework_version="1.8.0",                 # PyTorch framework version
    py_version="py36",                         # Python version for the PyTorch container
    entry_point="./entry_point.py",            # Path to the training script (relative to the source_dir or current dir)
    # source_dir='s3://...' or './source_directory' can be added if entry_point is within a directory.
)

# --- 3. Define a Training Step ---
# This step uses the configured estimator to run a training job.
step = TrainingStep(
    name="MyLocalTrainingStep",  # Name of the training step in the pipeline
    step_args=pytorch_estimator.fit( # The arguments for the step, typically from estimator.fit()
        inputs=TrainingInput( # Define the training data input
            s3_data="s3://amzn-s3-demo-bucket/my-data/train", # S3 URI for training data. For local mode,
                                                              # this can be a local file path: 'file:///path/to/local/data'
                                                              # or it might require localstack/moto if S3 access is emulated.
                                                              # Using a placeholder S3 URI as in the original example.
        ),
    ),
)

# --- 4. Define the Pipeline ---
# A pipeline consists of one or more steps.
pipeline = Pipeline(
    name="MyLocalPipeline",        # Name of the pipeline
    steps=[step],                 # List of steps included in this pipeline
    sagemaker_session=local_pipeline_session # Use the local session for the pipeline itself
)

# --- 5. Create (Upsert) the Pipeline Definition ---
# For local mode, this primarily registers the pipeline structure with the local session.
# In cloud mode, this would create/update the pipeline definition in SageMaker.
pipeline.create(
    role_arn=sagemaker.get_execution_role(), # IAM role for pipeline operations
    description="Example of a SageMaker pipeline running in local mode"
)

# --- 6. Start a Pipeline Execution (Locally) ---
# This command will trigger the pipeline to run on the local machine using Docker.
print("Starting pipeline execution in local mode...")
execution = pipeline.start()
print("Pipeline execution started. Waiting for completion...")
execution.wait() # Wait for the local execution to complete
print("Pipeline execution finished.")

# --- 7. Inspect Execution Results ---
# After execution, you can list the steps and their outcomes.
print("Listing execution steps...")
steps_summary = execution.list_steps()

# Example: Retrieve the name of the training job created by the TrainingStep
# The structure of 'steps_summary' can vary; adapt as needed based on actual output.
# This assumes the first step is the training step and extracts its job ARN/name.
# Note: For local mode, the 'Arn' might be a local identifier rather than a full AWS ARN.
try:
    # Accessing metadata for the first step (index 0)
    first_step_metadata = steps_summary[0] # More direct access if list_steps() returns a list of step summaries
    training_job_name = first_step_metadata.get('Metadata', {}).get('TrainingJob', {}).get('Arn')
    # Or, if it's a dictionary as in the original example:
    # training_job_name = steps_summary["PipelineExecutionSteps"][0]["Metadata"]["TrainingJob"]["Arn"]
    
    if training_job_name:
        print(f"Training job name/ARN from the first step: {training_job_name}")
        
        # Describe the training job to get more details (e.g., model artifacts path)
        # The 'TrainingJobName' for describe_training_job is usually the final part of the ARN.
        # For local mode, this might be the direct job name assigned by the local environment.
        actual_job_name = training_job_name.split('/')[-1] if '/' in training_job_name else training_job_name

        print(f"Describing training job: {actual_job_name}...")
        step_outputs = local_pipeline_session.sagemaker_client.describe_training_job(
            TrainingJobName=actual_job_name
        )
        print("Training job description:")
        import json
        print(json.dumps(step_outputs, indent=2, default=str)) # Pretty print the output
    else:
        print("Could not determine training job name from pipeline execution steps.")
except (IndexError, KeyError, TypeError) as e:
    print(f"Error accessing training job details from pipeline steps: {e}")
    print("Pipeline execution steps summary:")
    import json
    print(json.dumps(steps_summary, indent=2, default=str))
