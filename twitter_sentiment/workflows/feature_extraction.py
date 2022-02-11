from flytekit import workflow

@workflow
def my_wf() -> str:
    bucket_name = create_bucket(bucket_name='feature_store')



if __name__ == "__main__":
    print(f"Running my_wf() { my_wf() }")
