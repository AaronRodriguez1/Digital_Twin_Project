"""
Sagemaker_pipeline.py

Create a SageMaker pipeline for continuous training of the digital twin model.

"""

import boto3, sagemaker, os, time
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ModelStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

sess   = sagemaker.Session()
region = sess.boto_region_name
role   = os.environ["SAGEMAKER_EXEC_ROLE"]  
bucket = f"s3://{sess.default_bucket()}/digital-twin"


input_s3      = ParameterString("InputDataUrl",
                                default_value=f"{bucket}/data/latest/")
instance_type = ParameterString("TrainInstanceType",
                                default_value="ml.m5.xlarge")
epochs        = ParameterInteger("Epochs", default_value=3)


estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    role=role,
    framework_version="2.2",
    py_version="py311",
    hyperparameters={"epochs": epochs},
    instance_count=1,
    instance_type=instance_type,
    output_path=f"{bucket}/models",
    disable_profiler=True
)

train_step = TrainingStep(
    name="TrainDigitalTwin",
    estimator=estimator,
    inputs={"training": input_s3}
)

model_reg  = RegisterModel(
    name="RegisterDigitalTwin",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/plain"],
    response_types=["text/plain"],
    inference_instances=["ml.t3.medium"],
    transform_instances=["ml.m5.large"]
)

model_step = ModelStep(name="RegModel", step_args=model_reg)

pipeline = Pipeline(
    name="DigitalTwin-Continuous-Training",
    parameters=[input_s3, instance_type, epochs],
    steps=[train_step, model_step],
    sagemaker_session=sess
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Started pipeline execution: {execution.arn}")
