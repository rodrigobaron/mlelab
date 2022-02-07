from flytekit import task
import os
import re


@task(cache=True, cache_version="v1")
def create_bucket(
    bucket_name: str, 
    s3_access_key_id:str, 
    s3_secret_access_key:str, 
    s3_endpoint:str
) -> str:
    """Create a S3 buecket given the access_id, access_secret and endpoint.
    """
    client = boto3.client(
        "s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        use_ssl=False,
        endpoint_url=s3_endpoint,
    )

    try:
        client.create_bucket(Bucket=bucket_name)
    except client.exceptions.BucketAlreadyOwnedByYou:
        logger.info(f"Bucket {bucket_name} has already been created by you.")
        pass

    endpoint = re.sub(r"^(http|https)://?", "s3:///", url)
    return f"{endpoint}/{bucket_name}"
