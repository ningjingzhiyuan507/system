import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import month_of_year_index, day_of_week_index


COVARIATES_NUM = [
    '预约数2', '疫情', '疫情封控', '雨雪', '元旦', '春节', '清明节', '劳动节', '端午节',
    '中秋节', '国庆节'
]
TIME_FEATURES = time_features = time_features_from_frequency_str('D') + \
            [month_of_year_index, day_of_week_index]  # 月份和星期


def format_data(df, target_col, covariates, freq='12H',
                idx_start=None, idx_end=None):
    """
    Format data into DeepAR-compatible format.
    """
    if idx_start is None:
        idx_start, idx_end = 0, len(df)

    return ListDataset(
        [
            {
                FieldName.START: df['日期'].iloc[idx_start],
                FieldName.TARGET: df[target_col].iloc[idx_start:idx_end].values,
                FieldName.FEAT_DYNAMIC_REAL: df.iloc[idx_start:idx_end][covariates].values.T,
            }
        ],
        freq=freq,
    )


class DeepARModel:
    """训练DeepAR模型并滚动更新预测"""
    def __init__(self, best_params, freq='12H', prediction_length=2):
        self.best_params = best_params
        self.freq = freq
        self.prediction_length = prediction_length
        self.model = None

    def train(self, df_dev, target_col, covariates):
        dev_data = format_data(df_dev, target_col, covariates, self.freq)

        self.model = DeepAREstimator(
            prediction_length=self.prediction_length,
            freq=self.freq,
            batch_size=128,
            patience=30,
            num_feat_dynamic_real=len(covariates),
            time_features=TIME_FEATURES,
            trainer_kwargs={"max_epochs": 200},
            **self.best_params
        ).train(dev_data, num_workers=4)

    def predict(self, df, target_col, covariates, idx):
        """
        Singel-step prediction.
        :param df: 全部数据集
        :param target_col: 接诊数
        :param covariates: 协变量
        :param idx: 样本截止下标
        :return:
        """
        test_data = format_data(
            df, target_col, covariates, self.freq, self.prediction_length,
            idx_start=0, idx_end=idx+self.prediction_length)

        pred = list(self.model.predict(test_data))[0]
        return list(pred.quantile(0.5))

    def update(self, df_train, target_col, covariates, idx_end,
               context_lenght=14, update_window=56):
        """
        Update model with a smaller dataset.
        :param df_train:
        :param idx_end:
        :return:
        """
        idx_start = max(0, idx_end-update_window)
        update_data = format_data(
            df_train, target_col, covariates, self.freq, idx_start=idx_start,
            idx_end=idx_end)
        self.model = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=context_lenght,
            batch_size=32,
            num_feat_dynamic_real=len(covariates),
            time_features=TIME_FEATURES,
            patience=10,
            trainer_kwargs={"max_epochs": 30}
        ).train(update_data, network=self.model, num_workers=4)

    def rolling_predict(self, df, target_col, covariates, idx_start):
        """
        Rolling prediction with periodic updates.
        :param df:
        :param idx_start:
        :return:
        """
        preds = [None] * len(df)
        for i in range(idx_start, len(df) - 1, self.prediction_length):
            preds[i:i+self.prediction_length] = self.predict(
                df, target_col, covariates, i
            )
            if (i-idx_start) % 56 == 0:
                self.update(df, target_col, covariates, i)

        df['yhat'] = preds
        return df


class HyperparameterOptimizerDeepAR:
    """针对DeepAR进行超参数优化"""
    def __init__(self, model_class, param_space, n_split):
        self.model_class = model_class
        self.param_space = param_space
        self.n_split = n_split

    def objective(self, trial, df_dev, target_col, covariates):
        params = {key: dist(trial) for key, dist in self.param_space.items()}
        tscv = TimeSeriesSplit(n_splits=self.n_split)

        losses = []
        for train_idx, val_idx in tscv.split(df_dev):
            df_train, df_val = df_dev.iloc[train_idx], df_dev.iloc[val_idx]
            model = self.model_class(params)
            model.train(df_train, target_col, covariates)

            val_data = format_data(df_val, target_col, covariates)
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=val_data, predictor=model.model, num_samples=100
            )
            agg_metrics, _ = Evaluator()(list(ts_it), list(forecast_it))
            losses.append(agg_metrics['RMSE'])

        return np.mean(losses)

    def optimize(self, df_dev, target_col, covariates, n_trials=30):
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, df_dev, target_col, covariates),
            n_trials=n_trials
        )
        return study.best_trial.params


def data_process(file_path):
    """为DeepAR做数据准备"""
    df = pd.read_excel(file_path)
    cols_keep = ['日期', '接诊数'] + COVARIATES_NUM
    df = df[cols_keep]
    df['日期'] = pd.to_datetime(df['日期'])

    # 数据划分
    # condition_dev = (df['日期'] >= pd.to_datetime('2023-2-1')) & (df['日期'] < pd.to_datetime('2024-4-1'))
    condition_dev = df['日期'] < pd.to_datetime('2024-4-1')
    df_dev = df.loc[condition_dev, :].reset_index(drop=True)
    condition_test = df['日期'] >= pd.to_datetime('2024-4-1')
    df_test = df.loc[condition_test, :].reset_index(drop=True)
    return df_dev, df_test


if __name__ == '__main__':
    # 数据准备
    file_path = '/Users/gxz/Desktop/华山门诊/data/神内/神内-onehot.xlsx'
    df_dev, _ = data_process(file_path)

    # 超参数准备
    param_space = {
        'context_length': lambda trial: trial.suggest_int('context_length', 7, 28),
        'hidden_size': lambda trial: trial.suggest_int('hidden_size', 20, 128),
        'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
        'dropout_rate': lambda trial: trial.suggest_uniform('dropout_rate', 0.0, 0.5),
        'lr': lambda trial: trial.suggest_loguniform('lr', 1e-4, 1e-2),
    }
    optimizer = HyperparameterOptimizerDeepAR(DeepARModel, param_space, 5)
    best_params = optimizer.optimize(df_dev, '接诊数', COVARIATES_NUM)
