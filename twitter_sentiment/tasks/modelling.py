from flytekit import task



@task(cache_version="1.0", cache=True, limits=Resources(mem="200Mi"))
def split_traintest_dataset(
    df: pd.DataFrame, seed: int, test_split_ratio: float
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Retrieves the training dataset from the given blob location and then splits it using the split ratio and returns the result
    This splitter is only for the dataset that has the format as specified in the example csv. The last column is assumed to be
    the class and all other columns 0-8 the features.

    The data is returned as a schema, which gets converted to a parquet file in the back.
    """
    x = df[column_names[:8]]
    y = df[[column_names[-1]]]

    # split data into train and test sets
    return train_test_split(x, y, test_size=test_split_ratio, random_state=seed)


def serialize_model(model: BaseEstimator) -> JoblibSerializedFile:
    """Convert model object to compressed byte string."""
    out_file = "/tmp/model.joblib"
    with open(out_file, "wb") as f:
        joblib.dump(model, f, compress=True)
    return JoblibSerializedFile(path=out_file)


def deserialize_model(model_file: JoblibSerializedFile) -> BaseEstimator:
    """Load model object from compressed byte string."""
    with open(model_file, "rb") as f:
        model = joblib.load(f)
    return model


@task(cache=True, cache_version="v1")
def multinomial_naive_bayes(X_train_tfidf: pd.DataFrame,Y_train_tfidf:pd.DataFrame, alpha:float) -> JoblibSerializedFile:
    classfier_tfidf = MultinomialNB(alpha=alpha)
    classfier_tfidf.fit(X_train_tfidf,Y_train_tfidf)

    return serialize_model(classfier_tfidf)

