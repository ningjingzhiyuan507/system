"""华山门办皮肤科患者接诊数预测超参调优"""
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import month_of_year_index, day_of_week_index
import optuna
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import logging
import json
import os


# 定义交叉验证函数
def cross_validation(data, n_splits=2):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        yield train_data, test_data


# 定义Optuna优化目标函数
def objective(trial):
    context_length = trial.suggest_int("context_length", 7, 28)
    hidden_size = trial.suggest_int("hidden_size", 20, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    # 定义DeepAREstimator
    time_features = time_features_from_frequency_str("D") + [
        month_of_year_index, day_of_week_index]

    estimator = DeepAREstimator(
        prediction_length=2,
        context_length=context_length,
        freq="12H",
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        lr=lr,
        time_features=time_features,
        num_feat_dynamic_real=len(cols_covariate_num),
        batch_size=8,
        patience=30,
        trainer_kwargs={
            "max_epochs": 5
        }
    )

    losses = []
    for train_data, test_data in cross_validation(outpat_dev, n_splits=2):
        train_dataset = ListDataset(
            [{
                "start": train_data["日期"].iloc[0],
                "target": train_data["接诊数"].values,
                FieldName.FEAT_DYNAMIC_REAL: train_data[
                    cols_covariate_num].values.transpose(),
            }],
            freq="12H"
        )
        test_dataset = ListDataset(
            [{
                "start": test_data["日期"].iloc[0],
                "target": test_data["接诊数"].values,
                FieldName.FEAT_DYNAMIC_REAL: test_data[
                    cols_covariate_num].values.transpose(),
            }],
            freq="12H"
        )

        predictor = estimator.train(train_dataset, num_workers=4)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset, predictor=predictor, num_samples=100
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        evaluator = Evaluator()
        agg_metrics, _ = evaluator(tss, forecasts, num_series=len(test_dataset))
        losses.append(agg_metrics["RMSE"])

    return np.mean(losses)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    dir = os.path.abspath('.')
    outpat_dev = pd.read_excel(os.path.join(dir, 'outpat_dev.xlsx'))
    cols_covariate_num = [
        '预约量', '疫情', '疫情封控', '中雨', '风向_西风', '上下午',
        '元旦', '春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节', '寒假'
    ]

    # 运行Optuna优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1)

    with open(os.path.join(dir, 'best_params.txt'), 'w') as f:
        f.write(json.dumps(study.best_params))