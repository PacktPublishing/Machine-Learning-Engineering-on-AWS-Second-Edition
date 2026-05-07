import subprocess
import sys
import os
import argparse
import random
import string
import json

from contextlib import contextmanager


@contextmanager
def block(label):
    print(f"[{label}]: START")
    yield
    print(f"[{label}]: END")


def install_packages():
    with block("INSTALL_PACKAGES"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker==3.5.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker-serve==1.2.0"])


def generate_string(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def perform_fine_tuning(
    dataset,
    output_path,
    mlflow_experiment_name,
    mlflow_run_name,
    region,
    role_arn
):
    
    from sagemaker.train.sft_trainer import SFTTrainer
    from sagemaker.train.common import TrainingType
    from sagemaker.core.resources import ModelPackageGroup
    from sagemaker.serve import ModelBuilder

    model = "meta-textgeneration-llama-3-2-1b-instruct"

    with block("MODEL_PACKAGE_GROUP"):
        unique = generate_string()
        group_name = f"model-package-group-{unique}"
        model_package_group = ModelPackageGroup.create(
            model_package_group_name=group_name
        )

    with block("TRAINING"):
        trainer = SFTTrainer(
            model=model,
            training_type=TrainingType.LORA,
            model_package_group=model_package_group,
            training_dataset=dataset,
            s3_output_path=output_path,
            accept_eula=True,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_name=mlflow_run_name,
        )
        job = trainer.train(wait=True)
        print(f"Training job completed: {job.training_job_name}")

    with block("MODEL_BUILD"):
        model_builder = ModelBuilder(
            model=job,
            instance_type="ml.g5.4xlarge"
        )
        built_model = model_builder.build()

    with block("ARTIFACT_PROCESSING"):
        artifacts = job.__dict__["model_artifacts"]
        s3_model_artifacts = artifacts.__dict__["s3_model_artifacts"]

        merged = f"{s3_model_artifacts}/checkpoints/hf_merged"

        model_files_directory = f"/tmp/model_files_{unique}"
        os.makedirs(model_files_directory, exist_ok=True)

        run_cmd(f"aws s3 cp {merged} {model_files_directory}/ --recursive")

        tar_path = f"/tmp/model_{unique}.tar.gz"
        run_cmd(f"tar -czvf {tar_path} -C {model_files_directory} .")

        s3_bucket = output_path.replace("s3://", "").split("/")[0]
        s3_model_path = f"s3://{s3_bucket}/model/model-{unique}.tar.gz"

        run_cmd(f"aws s3 cp {tar_path} {s3_model_path}")

    with block("MODEL_CREATION"):
        containers = built_model.__dict__["containers"]
        container_image = containers[0].__dict__["image"]

        sm = boto3.client("sagemaker")
        model_name = f"model-{unique}"

        response = sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": container_image,
                "ModelDataUrl": s3_model_path,
                "Environment": {
                    "SHM_SIZE": "1g",
                    "HF_MODEL_ID": "/opt/ml/model",
                    "MAX_INPUT_LENGTH": "2048",
                    "MAX_TOTAL_TOKENS": "4096",
                    "NUM_SHARD": "1"
                }
            },
            ExecutionRoleArn=role_arn
        )

        model_arn = response["ModelArn"]

        print(f"Created model: {model_name}")
        print(f"Model ARN: {model_arn}")

    return {
        "region": region,
        "training_job_name": job.training_job_name,
        "training_job_arn": job.training_job_arn,
        "model_package_group_name": group_name,
        "model_package_group_arn": model_package_group.model_package_group_arn,
        "model_name": model_name,
        "model_arn": model_arn,
        "s3_model_artifacts": s3_model_artifacts,
        "s3_model_path": s3_model_path,
        "dataset": dataset,
        "output_path": output_path,
        "mlflow_experiment_name": mlflow_experiment_name,
        "mlflow_run_name": mlflow_run_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--mlflow-experiment-name", type=str, required=True)
    parser.add_argument("--mlflow-run-name", type=str, required=True)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--role-arn", type=str, required=True)

    args = parser.parse_args()

    with block("INSTALL"):
        install_packages()

    with block("REGION_SETUP"):
        region = args.region
        print(f"Using region: {region}")
        os.environ["AWS_DEFAULT_REGION"] = region
        import boto3
        boto3.setup_default_session(region_name=region)

    with block("FINE_TUNING_PIPELINE"):
        result = perform_fine_tuning(
            dataset=args.dataset,
            output_path=args.output_path,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            region=args.region,
            role_arn=args.role_arn
        )

    with block("WRITE_OUTPUT"):
        output_dir = "/opt/ml/processing/output"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "results.json")

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved results to {output_file}")
