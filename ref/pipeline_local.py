# https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-local-mode.html

from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline

local_pipeline_session = LocalPipelineSession()

pytorch_estimator = PyTorch(
    sagemaker_session=local_pipeline_session,
    role=sagemaker.get_execution_role(),
    instance_type="ml.c5.xlarge",
    instance_count=1,
    framework_version="1.8.0",
    py_version="py36",
    entry_point="./entry_point.py",
)

step = TrainingStep(
    name="MyTrainingStep",
    step_args=pytorch_estimator.fit(
        inputs=TrainingInput(s3_data="s3://amzn-s3-demo-bucket/my-data/train"),
    ),
)

pipeline = Pipeline(
    name="MyPipeline", steps=[step], sagemaker_session=local_pipeline_session
)

pipeline.create(
    role_arn=sagemaker.get_execution_role(), description="local pipeline example"
)

# pipeline will execute locally
execution = pipeline.start()

steps = execution.list_steps()

training_job_name = steps["PipelineExecutionSteps"][0]["Metadata"]["TrainingJob"]["Arn"]

step_outputs = pipeline_session.sagemaker_client.describe_training_job(
    TrainingJobName=training_job_name
)
