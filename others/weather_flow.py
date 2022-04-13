from metaflow import conda_base, FlowSpec, IncludeFile, Parameter, step, S3


def plot_prc(precisions, recalls, thresholds):
    import matplotlib.pyplot as plt

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


def download_data():
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "jsphyg/weather-dataset-rattle-package", path="data/weather", unzip=True
    )


@conda_base(
    python="3.8.10",
    libraries={
        "pandas": "1.3.4",
        "matplotlib": "3.5.1",
        "scikit-learn": "1.0.2",
        "seaborn": "0.11.2",
        "kaggle": "1.5.12"
    },
)
class WeatherFlow(FlowSpec):

    data_fname = Parameter(
        "data-path",
        help="The path to sst train file",
        default="data/weather/weatherAUS.csv",
    )

    @step
    def start(self):
        import pandas as pd

        df = pd.read_csv(self.data_fname)
        print(df.describe())

        self.df = df
        self.next(self.preprocess)

    @step
    def preprocess(self):
        import pandas as pd

        df = self.df
        zeros_cnt = df.isnull().sum().sort_values(ascending=False)
        percent_zeros = (df.isnull().sum() / df.isnull().count()).sort_values(
            ascending=False
        )

        missing_data = pd.concat(
            [zeros_cnt, percent_zeros], axis=1, keys=["Total", "Percent"]
        )
        print("# Missing data")
        print(missing_data)

        dropList = list(missing_data[missing_data["Percent"] > 0.15].index)
        dropList
        df.drop(dropList, axis=1, inplace=True)
        df["Location"].unique()
        self.df = df
        self.next(self.plot_dataframe)

    @step
    def plot_dataframe(self):
        import pandas as pd
        import seaborn as sns

        df = self.df
        sns.pairplot(df[:1000])

        df.head()
        df.drop(["Date"], axis=1, inplace=True)
        df.drop(["Location"], axis=1, inplace=True)

        ohe = pd.get_dummies(
            data=df, columns=["WindGustDir", "WindDir9am", "WindDir3pm"]
        )
        print(ohe.info())
        self.ohe = ohe
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        from sklearn import preprocessing
        from numpy import array
        from sklearn.model_selection import train_test_split

        df = self.df
        ohe = self.ohe

        ohe["RainToday"] = df["RainToday"].astype(str)
        ohe["RainTomorrow"] = df["RainTomorrow"].astype(str)

        lb = preprocessing.LabelBinarizer()

        ohe["RainToday"] = lb.fit_transform(ohe["RainToday"])
        ohe["RainTomorrow"] = lb.fit_transform(ohe["RainTomorrow"])

        ohe = ohe.dropna()
        y = ohe["RainTomorrow"]
        X = ohe.drop(["RainTomorrow"], axis=1)

        self.X, self.y = X, y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        self.next(self.train)

    @step
    def train(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        X_train, y_train = self.X_train, self.y_train

        print(X_train.info())

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "RFC",
                    RandomForestClassifier(
                        criterion="gini",
                        max_depth=10,
                        max_features="auto",
                        n_estimators=200,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        pipe.score(X_train, y_train)

        self.pipe = pipe
        self.next(self.evaluate)

    @step
    def evaluate(self):
        import matplotlib.pyplot as plt
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import precision_recall_curve

        X, y = self.X, self.y
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test
        pipe = self.pipe

        cross_val_score(pipe, X, y, cv=3)
        y_pred = pipe.predict(X_test)
        accuracy_score(y_test, y_pred)

        f1_score(y_test, y_pred)

        ns_probs = [0 for _ in range(len(y_test))]
        lr_probs = pipe.predict_proba(X_test)
        lr_probs = lr_probs[:, 1]

        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)

        print("No Skill: ROC AUC=%.3f" % (ns_auc))
        print("RFC: ROC AUC=%.3f" % (lr_auc))

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle="--", label="Dummy Classifer")
        plt.plot(lr_fpr, lr_tpr, marker=".", label="RFC")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.legend()
        # plt.show()

        y_scores = pipe.predict_proba(X_train)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        y_pred1 = (pipe.predict_proba(X_train)[:, 1] >= 0.8).astype(
            int
        )  # set threshold as 0.3
        precision_score(y_train, y_pred1)
        plot_prc(precisions, recalls, thresholds)
        self.next(self.end)

    @step
    def end(self):
        print("Weather next day forecast model complete!")


if __name__ == "__main__":
    download_data()
    WeatherFlow()
