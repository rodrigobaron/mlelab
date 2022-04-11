import os

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


def plot_prc (precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Thresholds')
    plt.legend(loc='center left')
    plt.ylim([0,1])


def download_data():
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('jsphyg/weather-dataset-rattle-package', path='data/weather', unzip=True)


@conda_base(
    python="3.8.10",
    libraries={"pandas": "1.3.4", "transformers": "4.12.3", "pytorch-gpu": "1.10.1", "pyyaml": "6.0.0"},
)
class WeatherNDFlow(FlowSpec):

    data_fname = Parameter(
        "data-path", help="The path to sst train file", default="data/weather/weatherAUS.csv"
    )

    @step
    def start(self):
        self.df = pd.read_csv(self.data_fname)
        print(self.df.describe())

    @step
    def pre_process(self):
        df = self.df
        zeros_cnt = df.isnull().sum().sort_values(ascending=False)
        percent_zeros = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)

        missing_data = pd.concat([zeros_cnt, percent_zeros], axis=1, keys=['Total', 'Percent'])
        print("# Missing data")
        print(missing_data)

        dropList = list(missing_data[missing_data['Percent'] > 0.15].index)
        dropList
        df.drop(dropList, axis=1, inplace=True)
        df['Location'].unique()
        self.df = df

    def plot_dataframe(self):
        df = self.df
        sns.pairplot(df[:1000])

        df.head()
        df.drop(['Date'], axis=1, inplace=True)
        df.drop(['Location'], axis=1, inplace=True)

        ohe = pd.get_dummies(data=df, columns=['WindGustDir','WindDir9am','WindDir3pm'])
        print(ohe.info())
        self.ohe = ohe

    def feature_engineering(self):
        from sklearn import preprocessing
        from numpy import array
        from sklearn.model_selection import train_test_split

        df = self.df
        ohe = self.ohe

        ohe['RainToday'] = df['RainToday'].astype(str)
        ohe['RainTomorrow'] = df['RainTomorrow'].astype(str)

        lb = preprocessing.LabelBinarizer()

        ohe['RainToday'] = lb.fit_transform(ohe['RainToday'])
        ohe['RainTomorrow'] = lb.fit_transform(ohe['RainTomorrow'])

        ohe = ohe.dropna()
        y = ohe['RainTomorrow']
        X = ohe.drop(['RainTomorrow'], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    def train(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        X_train.info()

        pipe = Pipeline([('scaler', StandardScaler()), ('RFC', RandomForestClassifier(criterion='gini', 
                                                                                    max_depth=10, 
                                                                                    max_features='auto',
                                                                                    n_estimators=200))])
                                                                                
        pipe.fit(X_train, y_train)
        pipe.score(X_train, y_train)

    def evaluate(self):
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import precision_recall_curve
        
        cross_val_score(pipe, X, y, cv=3)
        y_pred = pipe.predict(X_test)
        accuracy_score(y_test, y_pred)

        f1_score(y_test, y_pred)

        ns_probs = [0 for _ in range(len(y_test))]
        lr_probs = pipe.predict_proba(X_test)
        lr_probs = lr_probs[:, 1]

        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)

        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('RFC: ROC AUC=%.3f' % (lr_auc))

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Dummy Classifer')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='RFC')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend()
        plt.show()

        y_scores = pipe.predict_proba(X_train)[:,1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        y_pred1 = (pipe.predict_proba(X_train)[:,1] >= 0.8).astype(int) # set threshold as 0.3
        precision_score(y_train, y_pred1)
        plot_prc(precisions, recalls, thresholds)


if __name__ == "__main__":
    download_data()
    SSTFlow()
