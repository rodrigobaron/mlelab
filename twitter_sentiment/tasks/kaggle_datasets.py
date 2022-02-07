import kaggle


@task
def dowload_kaggle_dataset(dataset:str, path:str, unzip:bool = False) -> str:
    """Download kaggle dataset to some path.
    
    This is used to fetch datasets using kaggle api, it look at KAGGLE_USERNAME and KAGGLE_KEY
    environment variables to authenticate.
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=path, unzip=unzip)
    return path