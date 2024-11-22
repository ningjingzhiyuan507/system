"""
对回归预测值与真实值进行评价
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error


def predict_evaluate(y_true, yhat, y_true_0=True):
    """
    回归预测值评价
    :param y_true: Series, 真实值
    :param yhat: Series, 预测值
    :param y_true_0: True为删除真实值中为0的样本。MAPE指标中真实值不能为0
    :return:
    """
    mae = mean_absolute_error(y_true, yhat)  # 平均绝对误差
    medae = median_absolute_error(y_true, yhat)  # 中位绝对误差
    mse = mean_squared_error(y_true, yhat)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差/均方误差根
    r2 = r2_score(y_true, yhat)  # 拟合优度

    if y_true_0:
        y = pd.concat([y_true, yhat], axis=1)
        y = y.loc[y.iloc[:, 0] > 0, :].reset_index(drop=True)
        mape = mean_absolute_percentage_error(
            y.iloc[:, 0], y.iloc[:, 1])  # 平均绝对百分比误差
    else:
        mape = mean_absolute_percentage_error(y_true, yhat)

    return {'MAE': mae, 'MedAE': medae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse,
            'R2': r2}


if __name__ == '__main__':
    df = pd.read_excel('/Users/gxz/Desktop/华山门诊/result/神内/deepar/pred.xlsx')
    df = df.dropna()
    df_dev = df.loc[df['日期'] < pd.to_datetime('2024-4-1'), :]
    df_test = df.loc[df['日期'] >= pd.to_datetime('2024-4-1'), :]
    res1 = predict_evaluate(df_dev['接诊数'], df_dev['yhat'], y_true_0=F)
    res2 = predict_evaluate(df_test['接诊数'], df_test['yhat'])
    print(res1)
    print(res2)
