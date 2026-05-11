import subprocess
import sys
import os
import json
import argparse
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


def load_json(path):
    if path.startswith("s3://"):
        print(f"Detected S3 path: {path}")

        local_dir = "/tmp/json_downloads"
        os.makedirs(local_dir, exist_ok=True)

        filename = os.path.basename(path)
        local_path = os.path.join(local_dir, filename)
        cmd = f"aws s3 cp {path} {local_path}"

        print(f"Downloading from S3 using: {cmd}")

        subprocess.check_call(cmd, shell=True)
        path = local_path

        print(f"Downloaded to: {path}")

    else:
        print(f"Detected local path: {path}")

    with open(path, "r") as f:
        return json.load(f)


def run_evaluation(
    eval_dataset,
    custom_metric_path,
    output_path,
    evaluator_model,
    region
):

    from sagemaker.train.evaluate import LLMAsJudgeEvaluator
    import boto3

    with block("LOAD_RESULTS"):
        results_path = "/opt/ml/processing/input/results.json"
        results = load_json(results_path)
        print(results)

    with block("MLFLOW_CONFIG"):
        experiment_name = results["mlflow_experiment_name"]
        run_name = results["mlflow_run_name"] + "-eval"

    with block("CUSTOM_METRICS"):
        custom_metric = load_json(custom_metric_path)
        custom_metrics = json.dumps([custom_metric])
        print(custom_metric)

    with block("PATH_SETUP"):
        model_package_group = results["model_package_group_name"]

    with block("MODEL_PACKAGE_LOOKUP"):
        sm = boto3.client("sagemaker", region_name=region)

        response = sm.list_model_packages(
            ModelPackageGroupName=model_package_group,
            SortBy="CreationTime",
            SortOrder="Descending"
        )

        model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
        print("Model Package ARN:", model_package_arn)

    with block("EVALUATOR_INIT"):
        builtin_metrics = ["Completeness", "Faithfulness"]

        evaluator = LLMAsJudgeEvaluator(
            model=model_package_arn,
            evaluator_model=evaluator_model,
            dataset=eval_dataset,
            builtin_metrics=builtin_metrics,
            custom_metrics=custom_metrics,
            s3_output_path=output_path,
            evaluate_base_model=False,
            mlflow_experiment_name=experiment_name,
            mlflow_run_name=run_name
        )

        print(evaluator)

    with block("EXECUTION"):
        execution = evaluator.evaluate()
        print(execution)

    with block("WAIT"):
        execution.wait()

    with block("RESULTS"):
        execution.show_results(limit=10, offset=0, show_explanations=False)

    return {
        "model_package_arn": model_package_arn,
        "eval_output_path": output_path,
        "mlflow_experiment_name": experiment_name,
        "mlflow_run_name": run_name
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-dataset", type=str, required=True)
    parser.add_argument("--custom-metric-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--evaluator-model", type=str, default="amazon.nova-pro-v1:0")
    parser.add_argument("--region", type=str, required=True)

    args = parser.parse_args()

    with block("LIST_INPUT_DIRECTORY"):
        input_dir = "/opt/ml/processing/input"

        print(f"Listing contents of: {input_dir}")

        for root, dirs, files in os.walk(input_dir):
            print(f"\nDirectory: {root}")
            if dirs:
                print("Subdirectories:")
                for d in dirs:
                    print(f"  - {d}")

            if files:
                print("Files:")
                for f in files:
                    full_path = os.path.join(root, f)
                    print(f"  - {full_path}")


    with block("INSTALL"):
        install_packages()

    
    with block("REGION_SETUP"):
        region = args.region
        print(f"Using region: {region}")
        os.environ["AWS_DEFAULT_REGION"] = region
        import boto3
        boto3.setup_default_session(region_name=region)

    
    with block("EVALUATION_PIPELINE"):
        result = run_evaluation(
            eval_dataset=args.eval_dataset,
            custom_metric_path=args.custom_metric_path,
            output_path=args.output_path,
            evaluator_model=args.evaluator_model,
            region=args.region
        )

    
    with block("WRITE_OUTPUT"):
        output_dir = "/opt/ml/processing/output"
        input_dir = "/opt/ml/processing/input"

        os.makedirs(output_dir, exist_ok=True)

        input_results_path = os.path.join(input_dir, "results.json")
        eval_output_file = os.path.join(output_dir, "eval_results.json")
        copied_training_file = os.path.join(output_dir, "training_results.json")

        if os.path.exists(input_results_path):
            with open(input_results_path, "r") as src, open(copied_training_file, "w") as dst:
                dst.write(src.read())

        with open(eval_output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(eval_output_file)
        print(copied_training_file)
