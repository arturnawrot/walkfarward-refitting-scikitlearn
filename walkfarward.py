from cmath import e
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

TAKE_PROFIT = 0.001
STOP_LOSS = 0.0005

def get_X(data):
    """Return model design matrix X"""

    col = ['instrument', 'Timeframe', 'complete', 'Volume', 'Open', 'High', 'Low', 'Close', 'time']

    data = data.reset_index(drop=True)
    data = data[data.columns.difference(col)]
    data = data.values

    return data


def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(5).shift(-5) # Returns after roughly two days
    y[y.between(TAKE_PROFIT, -TAKE_PROFIT)] = 0 # Devalue returns smaller than 0.4%
    y[y > 0] = 1
    y[y < 0] = -1

    return y


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]

    return X, y

class Walkforward:

    def __init__(self, prediction_mask=None) -> None:
        self.models = []

        self.prediction_mask = prediction_mask

        self.is_fitted = False

    def raise_exception_if_models_are_empty(self):
        if len(self.models) == 0:
            raise Exception('self.models is empty')

    def add_model(self, clf, columns_to_train) -> None:
        self.models.append([clf, columns_to_train])

    def fit_models(self, data_to_train):
        self.raise_exception_if_models_are_empty()

        for index, item in enumerate(self.models):
            columns_to_train = self.models[index][1]

            X_train, y_train = get_clean_Xy(data_to_train[columns_to_train])

            if len(y_train) == 0:
                continue

            try:
                self.models[index][0].fit(X_train, y_train)
            except ValueError as e:
                if self.is_fitted == False:
                    raise e


        self.is_fitted = True

    def get_prediction(self, data_to_test) -> list:
        self.raise_exception_if_models_are_empty()

        models = self.models
        predictions_list = []
        

        for model in models:
            clf = model[0]
            columns_to_train = model[1]

            X = get_X(data_to_test[columns_to_train])

            try:
                res = clf.predict(X).tolist()
            except Exception:
                res = [0] * len(X)

            predictions_list.append(res)

        predictions_df = pd.DataFrame(predictions_list).T
        predictions_df['matching'] = predictions_df.eq(predictions_df.iloc[:, 0], axis=0).all(1).astype(int)
        predictions_df.loc[( (predictions_df['matching'] == 1) & (predictions_df[0] == -1) ), 'matching'] = -1

        return predictions_df['matching'].tolist()

    def get_walkfarward_prediction(self, data_to_test, N_TRAIN_SIZE, N_TRAIN_FREQUENCY) -> list:
        predictions = []
        
        for i in range(0, len(data_to_test.index)):
            if i < N_TRAIN_SIZE:
                predictions.extend([0])
                continue

            if i % N_TRAIN_FREQUENCY == 0:
                data_to_train = data_to_test[i-N_TRAIN_SIZE:i].loc[((data_to_test['signal'] == 1) | (data_to_test['signal'] == -1))] # go back N_TRAIN_SiZE

                self.fit_models(data_to_train)

                prediction = self.get_prediction(data_to_test)[i:i+N_TRAIN_FREQUENCY] # go ahead from i to N_TRAIN_FREQUENCY
                predictions.extend(prediction)

        if len(predictions) != len(data_to_test.index):
            missing = len(data_to_test.index) - len(predictions)

            prediction = self.get_prediction(data_to_test)[-missing:] # go ahead from i to N_TRAIN_FREQUENCY
            predictions.extend(prediction)


        if len(predictions) != len(data_to_test.index):
            raise Exception(f"Length of predictions ({len(predictions)}) is not the same as the length of the data_to_test ({len(data_to_test.index)})")

        return predictions
