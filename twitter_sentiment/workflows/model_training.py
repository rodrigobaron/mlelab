from flytekit import workflow
from twitter_sentiment.tasks.modelling import (
    split_traintest_dataset,
    multinomial_naive_bayes
)


@workflow
def split_twitter_dataset(df: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dfx_train, dfx_test, dfy_train, dfy_test = split_traintest_dataset(df, seed=42, test_split_ratio=0.2)

    return dfx_train, dfx_test, dfy_train, dfy_test

@workflow
def train_naive_bayes_tws(dfx: pd.DataFrame, dfy: pd.DataFrame) -> JoblibSerializedFile:
    model = multinomial_naive_bayes(dfx, dfy, alpha=0.3)
    return model
    


if __name__ == "__main__":
    print(f"Running my_wf() { my_wf() }")
