import boto3
import argparse
import sagemaker
import logging
import sys
import pandas as pd

from sagemaker.clarify import SageMakerClarifyProcessor, DataConfig, BiasConfig


logging_level = logging.getLevelName('INFO')
logging_stream_handler = logging.StreamHandler(sys.stdout)
logging_format = "%(asctime)s - %(message)s"
logging.basicConfig(
    level=logging_level,
    handlers=[logging_stream_handler],
    format=logging_format,
)
logger = logging.getLogger(__name__)


def log(log_string, logger=logger):
    logger.info(log_string)


def log_line(logger=logger):
    logger.info("█" * 50)


def log_section(log_string, logger=logger):
    log_line(logger=logger)
    log(log_string=log_string, logger=logger)
    log_line(logger=logger)


def get_session():
    return sagemaker.Session()


def get_bucket():
    return get_session().default_bucket()


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str)
    parser.add_argument('--data', type=str)
    
    arguments, _ = parser.parse_known_args()
    
    return arguments


def main():
    log_section("PROCESS ARGS")
    args = process_args()
    input_path = args.data
    role = args.role
    df = pd.read_csv(input_path)
    
    log_section("UPLOAD FILE TO S3")
    sagemaker_session = get_session()
    bucket = get_bucket()
    dataset_s3_uri = sagemaker_session.upload_data(
        path=input_path,
        bucket=bucket,
        key_prefix="clarify-pretraining-bias"
    )

    log_section("INITIALIZE AND CONFIGURE CLARIFY PROCESSOR")
    clarify_processor = SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session
    )

    data_config = DataConfig(
        s3_data_input_path=dataset_s3_uri,
        s3_output_path=f"s3://{bucket}/output",
        label='predicted_label',
        headers=df.columns.to_list(),
        dataset_type='text/csv'
    )

    bias_config = BiasConfig(
        label_values_or_threshold=[1], 
        facet_name='sex'
    )

    log_section("RUN CLARIFY PROCESSOR")
    clarify_processor.run_pre_training_bias(data_config=data_config, data_bias_config=bias_config, methods=['CI'])

    dest = clarify_processor.latest_job.outputs[0].destination
    print(dest)


main()
