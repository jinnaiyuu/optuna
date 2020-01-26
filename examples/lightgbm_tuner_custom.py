"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM tuner.

In this example, we optimize the validation accuracy of cancer detection.
We optimize both the choice of booster model and their hyperparameters.

You can execute this code directly.
    $ python lightgbm_tuner_simple.py

"""

import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb_original
import optuna.integration.lightgbm as lgb


def my_accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'my_accuracy', np.mean(labels == (preds > 0.5)), True

if __name__ == '__main__':
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb_original.Dataset(train_x, label=train_y)
    dval = lgb_original.Dataset(val_x, label=val_y)

    params = {
        'objective': 'binary',
        'metric': 'my_accuracy',
        'verbosity': -1,
        'boosting_type': 'gbdt',
    }

    best_params, tuning_history = dict(), list()

    model = lgb.train(params,
                      dtrain,
                      best_params=best_params,
                      valid_sets=[dtrain, dval],
                      feval=my_accuracy,
                      verbose_eval=100,
                      early_stopping_rounds=100,
                      )

    prediction = np.rint(model.predict(val_x, num_iteration=model.best_iteration))
    accuracy = accuracy_score(val_y, prediction)

    print('Number of finished trials: {}'.format(len(tuning_history)))
    print('Best params:', best_params)
    print('  Accuracy = {}'.format(accuracy))
    print('  Params: ')
    for key, value in best_params.items():
        print('    {}: {}'.format(key, value))
