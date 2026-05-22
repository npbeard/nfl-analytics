"""
AWS cloud pipeline reference implementation.

Shows how each layer of the proposed architecture maps to an AWS service.
This module is structured so the same logic that runs locally can be
wired to real AWS resources by swapping the clients below.

Proposed architecture (as discussed in group meeting 2026-05-22):
  Ingestion  : S3 (batch files) + Kinesis Data Streams (live NGS)
  Processing : AWS Glue (ETL/feature engineering)
  Storage    : S3 Data Lake (raw/processed/curated zones) + Redshift (analytics)
  ML         : SageMaker (training + real-time inference endpoint)
  AI/NLP     : Amazon Bedrock (Claude) via chatbot interface
  Serving    : API Gateway + Lambda
  Monitoring : CloudWatch + Glue Data Quality
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AWSConfig:
    region: str = "eu-west-1"
    raw_bucket: str = "nfl-data-lake-raw"
    processed_bucket: str = "nfl-data-lake-processed"
    curated_bucket: str = "nfl-data-lake-curated"
    kinesis_stream: str = "nfl-ngs-live-stream"
    glue_database: str = "nfl_analytics"
    redshift_cluster: str = "nfl-redshift-cluster"
    sagemaker_endpoint: str = "nfl-injury-prediction-v1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    environment: str = "prod"  # dev | pre-prod | prod  (R2)


# ---------------------------------------------------------------------------
# S3 — batch ingestion (Stadium FTP files, weather API dumps)
# ---------------------------------------------------------------------------

class S3Ingestor:
    """
    Handles batch uploads to the raw S3 zone.
    In production: triggered by EventBridge on a schedule or S3 event notification.
    """

    def __init__(self, config: AWSConfig):
        self.config = config

    def upload_play_list(self, local_path: str) -> str:
        s3_key = f"raw/plays/{local_path.split('/')[-1]}"
        s3_uri = f"s3://{self.config.raw_bucket}/{s3_key}"
        logger.info("Would upload %s → %s", local_path, s3_uri)
        # boto3: self._client.upload_file(local_path, self.config.raw_bucket, s3_key)
        return s3_uri

    def upload_injury_records(self, local_path: str) -> str:
        s3_key = f"raw/injuries/{local_path.split('/')[-1]}"
        s3_uri = f"s3://{self.config.raw_bucket}/{s3_key}"
        logger.info("Would upload %s → %s", local_path, s3_uri)
        return s3_uri


# ---------------------------------------------------------------------------
# Kinesis — real-time NGS sensor stream
# ---------------------------------------------------------------------------

class KinesisNGSProducer:
    """
    Publishes live player-tracking frames to Kinesis Data Streams.
    One record per sensor frame; partition key = PlayKey for ordering.
    Downstream: Kinesis Data Analytics (Flink) → S3 / Redshift.
    """

    def __init__(self, config: AWSConfig):
        self.config = config

    def put_tracking_frame(self, frame: dict[str, Any]) -> dict:
        record = {
            "StreamName": self.config.kinesis_stream,
            "Data": json.dumps(frame).encode(),
            "PartitionKey": str(frame.get("PlayKey", "unknown")),
        }
        logger.info("Would put Kinesis record: PlayKey=%s time=%.1f",
                    frame.get("PlayKey"), frame.get("time", 0))
        # boto3: self._client.put_record(**record)
        return {"ShardId": "shardId-000000000000", "SequenceNumber": "simulated"}

    def stream_tracking_dataframe(self, df) -> int:
        """Iterate a tracking DataFrame and publish each row."""
        sent = 0
        for _, row in df.iterrows():
            self.put_tracking_frame(row.to_dict())
            sent += 1
        return sent


# ---------------------------------------------------------------------------
# Glue — ETL & feature engineering
# ---------------------------------------------------------------------------

class GlueETLJob:
    """
    Represents the Glue job that runs feature_engineering.py at scale.
    Triggered after new files land in the raw S3 zone.
    Output is written to the processed zone and catalogued in the Glue Data Catalog.
    """

    JOB_NAME = "nfl-injury-feature-engineering"

    def __init__(self, config: AWSConfig):
        self.config = config

    def start(self, input_s3_uri: str, output_s3_uri: str) -> str:
        job_run_id = "jr_simulated_abc123"
        logger.info(
            "Would start Glue job '%s': %s → %s (run_id=%s)",
            self.JOB_NAME, input_s3_uri, output_s3_uri, job_run_id,
        )
        # boto3: self._client.start_job_run(JobName=self.JOB_NAME, Arguments={...})
        return job_run_id

    def get_data_quality_report(self, table: str) -> dict:
        """CloudWatch + Glue Data Quality rule evaluation (R6)."""
        return {
            "table": table,
            "rules_passed": 12,
            "rules_failed": 0,
            "completeness": 0.997,
            "uniqueness_PlayKey": 1.0,
        }


# ---------------------------------------------------------------------------
# SageMaker — injury prediction inference
# ---------------------------------------------------------------------------

class SageMakerPredictor:
    """
    Calls the deployed SageMaker real-time endpoint.
    The endpoint serves the RandomForest model trained in src/ml/injury_prediction.py.
    """

    def __init__(self, config: AWSConfig):
        self.config = config

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """
        Returns probability of each injury severity threshold.
        In production: invokes the SageMaker endpoint via boto3 sagemaker-runtime.
        """
        payload = json.dumps({"features": features})
        logger.info("Would invoke endpoint '%s' with %d features",
                    self.config.sagemaker_endpoint, len(features))
        # Simulated response shape — real endpoint returns these probabilities
        return {
            "DM_M1_prob":  0.82,
            "DM_M7_prob":  0.61,
            "DM_M28_prob": 0.34,
            "DM_M42_prob": 0.18,
        }

    def batch_transform(self, input_s3_uri: str, output_s3_uri: str) -> str:
        job_name = "nfl-injury-batch-2026"
        logger.info("Would start SageMaker batch transform: %s → %s", input_s3_uri, output_s3_uri)
        # boto3: self._client.create_transform_job(...)
        return job_name


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

class NFLDataPipeline:
    """
    Ties together the full batch pipeline:
      Upload → Glue ETL → SageMaker batch predictions → Redshift
    Meant to be triggered daily by EventBridge after game data arrives.
    """

    def __init__(self, config: AWSConfig | None = None):
        self.config = config or AWSConfig()
        self.s3 = S3Ingestor(self.config)
        self.glue = GlueETLJob(self.config)
        self.sagemaker = SageMakerPredictor(self.config)

    def run_daily_batch(self, plays_path: str, injuries_path: str) -> dict:
        print(f"[Pipeline] Environment: {self.config.environment}")

        # 1. Ingest
        plays_uri = self.s3.upload_play_list(plays_path)
        inj_uri = self.s3.upload_injury_records(injuries_path)
        print(f"[Pipeline] Uploaded raw files to S3")

        # 2. ETL
        processed_uri = f"s3://{self.config.processed_bucket}/injury_features/latest/"
        run_id = self.glue.start(plays_uri, processed_uri)
        print(f"[Pipeline] Glue job started: {run_id}")

        # 3. Data quality check (R6)
        dq = self.glue.get_data_quality_report("injury_features")
        print(f"[Pipeline] Data quality: {dq['rules_passed']} rules passed, "
              f"completeness={dq['completeness']:.1%}")

        # 4. Batch inference
        predictions_uri = f"s3://{self.config.curated_bucket}/injury_predictions/latest/"
        batch_job = self.sagemaker.batch_transform(processed_uri, predictions_uri)
        print(f"[Pipeline] SageMaker batch job: {batch_job}")

        return {
            "status": "success",
            "glue_run_id": run_id,
            "predictions_uri": predictions_uri,
            "data_quality": dq,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pipeline = NFLDataPipeline(AWSConfig(environment="dev"))
    result = pipeline.run_daily_batch(
        plays_path="PlayList.csv",
        injuries_path="InjuryRecord.csv",
    )
    print("\nPipeline result:")
    print(json.dumps(result, indent=2))

    # Demo: real-time inference for a single play
    predictor = SageMakerPredictor(AWSConfig())
    sample_features = {
        "is_synthetic": 1,
        "field_synthetic": 1,
        "is_indoor": 0,
        "Temperature": 68.0,
        "PlayerDay": 45,
        "PlayerGamePlay": 32,
        "play_Pass": 1,
        "play_Rush": 0,
        "body_Knee": 1,
    }
    probs = predictor.predict(sample_features)
    print("\nReal-time injury severity prediction:")
    for k, v in probs.items():
        print(f"  {k}: {v:.0%}")
