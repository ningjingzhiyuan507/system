"""构建模型对测试集进行滚动重新训练然后预测"""
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import month_of_year_index, day_of_week_index
import pandas as pd
import os
import torch
import logging


def df_process(df):
    df['上下午'] = df['上下午'].replace({'上午': '00:00:00', '下午': '12:00:00'})
    df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
    df['日期'] = df['日期'] + ' ' + df['上下午']
    df['上下午'] = df['上下午'].replace({'00:00:00': 0, '12:00:00': 1})

    cols_keep = [  # DeepAR内含季度趋势
        '日期', '上下午', '接诊数', '预约量',
        '元旦', '春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节', '寒假',
        '疫情', '疫情封控', '中雨', '风向_西风', 'dataset_col'
    ]
    df = df.loc[:, cols_keep]

    return df


class RollingUpdatePredict:
    def __init__(self, best_params, cols_cov_num):
        self.best_params = best_params
        self.cols_cov_num = cols_cov_num
        self.model = None

    def parepare_ts(self, df, idx_end, test_ts=False):
        pat = 1 if test_ts else -1  # 测试集需要将预测那天的协变量信息输入
        data_ts = ListDataset(
            [
                {
                    FieldName.START: df['日期'][0],
                    FieldName.TARGET: df['接诊数'][0:idx_end],
                    FieldName.FEAT_DYNAMIC_REAL: df.loc[0:idx_end + pat,
                                                 self.cols_cov_num].values.transpose()
                }
            ],
            freq='12H'
        )

        return data_ts

    def train_model(self, df, idx_end):
        train_data = self.parepare_ts(df, idx_end)

        time_features = time_features_from_frequency_str("D") + [
            month_of_year_index, day_of_week_index]

        deepar = DeepAREstimator(
            freq='12H', prediction_length=2, batch_size=128,
            num_feat_dynamic_real=len(self.cols_cov_num),
            time_features=time_features,
            patience=15,
            num_batches_per_epoch=50,
            **self.best_params,
            trainer_kwargs={
                "max_epochs": 100,
                "devices": 1 if torch.cuda.is_available() else None
            }
        )
        deepar = deepar.train(train_data, num_workers=4)

        self.model = deepar

    def predict(self, df, idx_end):
        test_data = self.parepare_ts(df, idx_end, test_ts=True)

        pred = list(self.model.predict(test_data))
        pred_value = pred[0].quantile(0.5)

        return list(pred_value)

    def rolling_predict(self, df, idx_start):
        df_cnt = df.shape[0]
        preds = [None] * df_cnt
        for i in range(idx_start, df_cnt - 1, 2):
            self.train_model(df, i)

            pred = self.predict(df, i)
            preds[i:(i + 2)] = pred

        df_copy = df.copy()
        df_copy['yhat'] = pd.Series(preds)
        return df_copy


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
    logging.info('===== START =====')

    dir = os.path.abspath('.')
    outpatients = pd.read_excel(os.path.join(dir, 'outpatients-onehot.xlsx'))
    outpatients = df_process(outpatients)

    cols_covariate_num = [
        '预约量', '疫情', '疫情封控', '中雨', '风向_西风', '上下午',
        '元旦', '春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节', '寒假'
    ]
    idx_start = 3202  # 测试集开始的行数
    best_params = {
        "context_length": 20,
        "hidden_size": 72,
        "num_layers": 2,
        "dropout_rate": 0.20955705762745974,
        "lr": 0.002735580676907022}

    ins = RollingUpdatePredict(best_params, cols_covariate_num)

    test_preds = ins.rolling_predict(outpatients, idx_start)
    test_preds = test_preds.loc[test_preds['dataset_col'] == 'test', :]
    test_preds.to_excel(os.path.join(dir, 'test_preds.xlsx'), index=False)

    logging.info('===== everthing OK =====')