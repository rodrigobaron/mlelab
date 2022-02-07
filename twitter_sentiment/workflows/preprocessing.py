import os
import typing


from flytekit import task, workflow
from twitter_sentiment.tasks.s3_store import create_bucket
from twitter_sentiment.tasks.kaggle_datasets import create_bucket

from twitter_sentiment.tasks.feature_engineering import (
    load_dataset,
)


@workflow
def my_wf() -> str:
    s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
    s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
    s3_endpoint = os.getenv("S3_ENDPOINT")

    bucket_path = create_bucket(
        'datasets', 
        s3_access_key_id, 
        s3_secret_access_key, 
        s3_endpoint
    )

    bucket_path = dowload_kaggle_dataset(
        "twitter-sentiment-dataset",
        bucket_path, 
        unzip=True
    )

    df = load_dataset(bucket_path, ['Twitter_Data.csv'])




if __name__ == "__main__":
    print(f"Running my_wf() { my_wf() }")
