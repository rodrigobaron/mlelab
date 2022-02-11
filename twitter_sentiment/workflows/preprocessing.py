import sys
sys.path.append("./")

import os
from typing import TypeVar

import boto3
import re
import logging
import kaggle

from flytekit import task, kwtypes, workflow
from flytekit.types.file import FlyteFile

import pandas as pd

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated


training_cols = kwtypes(clean_text=str, category=int)
testing_cols = kwtypes(clean_text=str)

DFTwitterSentiment = Annotated[pd.DataFrame, training_cols]
DFTwitter = Annotated[pd.DataFrame, testing_cols]


@task(cache=True, cache_version="v1.0")
def create_bucket(bucket_name: str) -> str:
    """Create a S3 buecket given the access_id, access_secret and endpoint.
    """
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        use_ssl=False,
        endpoint_url=os.getenv("S3_ENDPOINT"),
    )

    try:
        client.create_bucket(Bucket=bucket_name)
    except client.exceptions.BucketAlreadyOwnedByYou:
        logging.info(f"Bucket {bucket_name} has already been created by you.")
        pass

    return bucket_name

@task
def download_kaggle_dataset(dataset:str) -> FlyteFile[TypeVar("zip")]:
    """Download kaggle dataset to some path.
    
    This is used to fetch datasets using kaggle api, it look at KAGGLE_USERNAME and KAGGLE_KEY
    environment variables to authenticate.
    """
    kaggle.api.authenticate()
    tmp_dataset_path = "/tmp/datasets"
    dataset_path = f'{tmp_dataset_path}/{dataset.split("/")[-1]}.zip'
    kaggle.api.dataset_download_files(dataset, path=tmp_dataset_path, unzip=False)
    return dataset_path


@task(cache=False, cache_version="v1")
def load_twitter_sentiment_df(dataset_path:FlyteFile[TypeVar("zip")]) -> DFTwitterSentiment:
    return pd.read_csv(dataset_path)





#-----------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


@task(cache=True, cache_version="v1.0")
def text_stemming(df: pd.DataFrame) -> pd.DataFrame:
    # nltk.download('stopwords')
    ps = PorterStemmer()

    corpus = []
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', str(df["clean_text"][i]))
        review = review.lower().split()
        # Stemming
        stemmed = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(stemmed)
        corpus.append(review)

    df["clean_text"] = corpus
    df = df.dropna()
    df = df.reset_index()
    
    return df


@task(cache=True, cache_version="v1.0")
def term_frequency_vectorizer(df: pd.DataFrame) -> pd.DataFrame:
    df = df[:5000]
    tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
    X_tfidf = tfidf.fit_transform(df["clean_text"]).toarray()

    Y = df["category"]
    df_tfidf = pd.DataFrame(X_tfidf,columns = tfidf.get_feature_names())
    df_tfidf["output"] = Y
    return df_tfidf


@workflow
def load_twitter_sentiment_dataset() -> pd.DataFrame:
    # s3_access_key_id = "GV97T0GPQFVBTO4QWT6S" #os.getenv("S3_ACCESS_KEY_ID")
    # s3_secret_access_key = "91RYqgQu+HAeZh+HXe+kRZimVSGCQ00fgCkOF6O5" #os.getenv("S3_SECRET_ACCESS_KEY")
    # s3_endpoint = "http://192.168.0.222:30084" # os.getenv("S3_ENDPOINT")

    dataset_path = download_kaggle_dataset(
        dataset="saurabhshahane/twitter-sentiment-dataset",
    )

    df = load_twitter_sentiment_df(dataset_path=dataset_path)
    df = text_stemming(df=df)
    return term_frequency_vectorizer(df=df)
    # return df

# @workflow
# def twitter_feature_engineering(df:pd.DataFrame) -> pd.DataFrame:

#     df = text_stemming(df=df)
#     return term_frequency_vectorizer(df=df)


if __name__ == "__main__":
    df = load_twitter_sentiment_dataset()
    # df = twitter_feature_engineering(df)
    print(df.head(5))
