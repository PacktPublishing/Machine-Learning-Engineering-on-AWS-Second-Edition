import argparse
import sagemaker
import logging
import sys
import pandas as pd

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.huggingface.processing import HuggingFaceProcessor


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


def get_bucket():
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()

    return bucket


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str)
    parser.add_argument('--script', type=str, default="scripts/token_counter.py")
    parser.add_argument('--data', type=str, default="input/input.csv")
    parser.add_argument('--output', type=str, default="output/output.csv")
    parser.add_argument('--batch_size', type=str, default="10")
    
    arguments, _ = parser.parse_known_args()
    log(str(arguments))
    
    return arguments


def main():
    log_section("PROCESS ARGS")
    args = process_args()
    input_file = args.data
    output_file = args.output
    batch_size = args.batch_size
    role = args.role
    code_filename = args.script
    
    
    log_section("INITIALIZE PROCESSOR") 
    bucket = get_bucket()
    inputs = [ProcessingInput(source=input_file, destination="/opt/ml/processing/input")]
    outputs = [ProcessingOutput(output_name = output_file, source="/opt/ml/processing/output", destination=f's3://{bucket}/mlengineering/output/')]
    arguments = ["--batch_size", batch_size]
    instance_type = 'ml.p3.2xlarge'

    processor = HuggingFaceProcessor(
        role=role, 
        instance_type=instance_type, 
        transformers_version='4.6', 
        pytorch_version='1.8', 
        instance_count=1
    )

    log_section("RUN PROCESSOR")
    processor.run(code=code_filename, inputs=inputs, outputs=outputs, arguments=arguments, wait=True)

    log_section("CHECK OUTPUT")
    output = processor.latest_job.outputs[0]
    destination = output.destination
    output_path = destination + "output.csv"

    output_df = pd.read_csv(output_path, header=None)
    print(output_df)


main()
