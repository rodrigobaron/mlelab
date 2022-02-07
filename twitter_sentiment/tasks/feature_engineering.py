from flytekit import task
from typing import List
import pandas as pd


@task(cache=True, cache_version="v1")
def load_dataset(base_path:str, files: List[str]) -> pd.DataFrame:
    filepaths = [f"{base_path}/{f}" for f in files]
    return pd.concat(map(pd.read_csv, filepaths))


@task
def text_stemming(df: pd.DataFrame) -> pd.DataFrame:
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


@task
def term_frequency_vectorizer(df: pd.DataFrame) -> pd.DataFrame:
    tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
    X_tfidf = tfidf.fit_transform(df["clean_text"]).toarray()

    Y = df_stemmed["category"]
    df_tfidf = pd.DataFrame(X_tfidf,columns = tfidf.get_feature_names())
    df_tfidf["output"] = Y
    
    return df_tfidf
