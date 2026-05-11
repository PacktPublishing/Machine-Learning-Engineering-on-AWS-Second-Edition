import boto3


class PipelineExecution:
    def __init__(self, execution):
        self.execution = execution
        self.sm = boto3.client("sagemaker")
        self.logs_client = boto3.client("logs")

        raw_steps = execution.list_steps()

        def sort_key(step):
            start = step.get("StartTime")
            return start if start is not None else ""

        self.steps_metadata = sorted(raw_steps, key=sort_key)

    
    def number_of_steps(self):
        return len(self.steps_metadata)

    def step(self, index: int):
        if index < 1 or index > len(self.steps_metadata):
            raise IndexError(f"Step index out of range (1-{len(self.steps_metadata)})")
        return PipelineStep(self.steps_metadata[index - 1], self.sm, self.logs_client)

    
    def outputs(self):
        if not self.steps_metadata:
            return []
        return self.step(len(self.steps_metadata)).outputs()


class PipelineStep:
    def __init__(self, step_metadata, sm_client, logs_client):
        self.metadata = step_metadata
        self.sm = sm_client
        self.logs_client = logs_client

        self.name = step_metadata.get("StepName")
        self.job_type, self.arn = self._detect_job()

        self._desc_cache = None  

    def _detect_job(self):
        metadata = self.metadata.get("Metadata", {})

        if "ProcessingJob" in metadata:
            return "processing", metadata["ProcessingJob"]["Arn"]

        if "TrainingJob" in metadata:
            return "training", metadata["TrainingJob"]["Arn"]

        raise ValueError(f"Unsupported step type for step: {self.name}")

    def _get_job_name(self):
        return self.arn.split("/")[-1]

    def _describe(self):
        if self._desc_cache:
            return self._desc_cache

        job_name = self._get_job_name()

        if self.job_type == "processing":
            self._desc_cache = self.sm.describe_processing_job(
                ProcessingJobName=job_name
            )
        elif self.job_type == "training":
            self._desc_cache = self.sm.describe_training_job(
                TrainingJobName=job_name
            )

        return self._desc_cache

    
    def logs(self, tail=False):
        job_name = self._get_job_name()

        log_group = (
            "/aws/sagemaker/ProcessingJobs"
            if self.job_type == "processing"
            else "/aws/sagemaker/TrainingJobs"
        )

        streams = self.logs_client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name
        )

        for stream in streams.get("logStreams", []):
            stream_name = stream["logStreamName"]

            events = self.logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                startFromHead=not tail
            )

            for event in events["events"]:
                print(event["message"])

    
    def inputs(self):
        desc = self._describe()

        if self.job_type == "processing":
            return [
                inp["S3Input"]["S3Uri"]
                for inp in desc.get("ProcessingInputs", [])
                if "S3Input" in inp
            ]

        if self.job_type == "training":
            return [
                inp["DataSource"]["S3DataSource"]["S3Uri"]
                for inp in desc.get("InputDataConfig", [])
            ]

        return []

    
    def outputs(self):
        desc = self._describe()

        if self.job_type == "processing":
            return [
                out["S3Output"]["S3Uri"]
                for out in desc.get("ProcessingOutputConfig", {}).get("Outputs", [])
            ]

        if self.job_type == "training":
            return [desc["OutputDataConfig"]["S3OutputPath"]]

        return []
