from causica.lightning.modules.deci_module import DECIModule
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.distributions import ContinuousNoiseDist
from causica.datasets.causica_dataset_format import Variable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch

import pandas as pd
import numpy as np
from operator import itemgetter
import warnings
import os

warnings.filterwarnings("ignore")
test_run = bool(os.environ.get("TEST_RUN", False))  # used by testing to run the notebook as a script


def get_variable_meta(df, path_dytpes_config, colnames_map):
    """
    构建符合deci的数据类型配置信息
    Args:
        df: 数据集
        path_dytpes_config: 数据类型配置文件路径
        colnames_map: 不满足deci要求的列名映射

    Returns: deci的meta

    """
    dtypes = pd.read_excel(
        os.path.join(path_dytpes_config, '字段缺失情况.xlsx'),
        usecols=['字段', '数据类型'])
    dtypes = dtypes.dropna()

    variables_metadata = []
    for _, row in dtypes.iterrows():
        field, dtype = row['字段'], row['数据类型']
        if dtype not in ['分类', '连续']:
            continue

        col_type = {}
        field_new = colnames_map[field] if field in colnames_map else field
        col_type['name'] = field_new
        col_type['group_name'] = field_new

        if dtype == '分类':
            if len(df[field].dropna().unique()) == 2:
                col_type['type'] = 'binary'
            else:
                col_type['type'] = 'categorical'
        elif dtype == '连续':
            col_type['type'] = 'continuous'
        variables_metadata.append(col_type)

    if len(variables_metadata) != df.shape[1]:
        print("数据类型数量和数据集的列数对不上")
    return variables_metadata


def set_constraint_matrix(data_module, cols_independent, cols_dependent,
                          ind_dep_supp, other_detail):
    """
    构建约束矩阵
    cols_independent: 自变量，没有节点指向这些节点。list；
    cols_dependent: 因变量，这些节点不能指向任何节点。list；
    ind_dep_supp: 对自变量和因变量的修正，某些可能误杀。dict,key可指向value；
    other_detail: 其它具体的约束条件，key不能指向value, {key: [1, 2]}
    """
    name_to_idx = {key: i for i, key in
                   enumerate(data_module.dataset_train.keys())}
    nodes_cnt = len(name_to_idx)
    constraint_matrix = np.full((nodes_cnt, nodes_cnt), np.nan,
                                dtype=np.float32)

    # 自变量 & 因变量
    independent_idx = itemgetter(*cols_independent)(name_to_idx)
    constraint_matrix[:, independent_idx] = 0.0

    dependent_idx = itemgetter(*cols_dependent)(name_to_idx)
    constraint_matrix[dependent_idx, :] = 0.0

    for key, val in ind_dep_supp.items():
        constraint_matrix[name_to_idx[key], name_to_idx[val]] = np.nan

    # 具体对约束
    for key, vals in other_detail.items():
        for v in vals:
            constraint_matrix[name_to_idx[key], name_to_idx[v]] = 0.0

    return constraint_matrix


if __name__ == '__main__':
    df = pd.read_csv(
        '/Users/gxz/Desktop/PT/因果发现/data/process/integration_fillna.csv')
    df = df.drop(columns=['json_name', '时间'])

    # 数据准备
    colnames_map = {'电解质-CA++(7.4)': '电解质-CA++'}
    variables_metadata = get_variable_meta(
        df, '/Users/gxz/Desktop/PT/因果发现/data', colnames_map)

    data_module = BasicDECIDataModule(
        df,
        variables=[Variable.from_dict(d) for d in variables_metadata],
        batch_size=512,
        normalize=True
    )

    cols_independent = ["年龄", "性别", "婚姻", "是否生育", "籍贯"]
    cols_dependent = ['outcome']
    ind_dep_supp = {
        '年龄': '婚姻',
        '年龄': '是否生育',
        '婚姻': '是否生育',
    }
    cols_diag = [
        '诊断分类_主动脉疾病', '诊断分类_先天性心脏病', '诊断分类_先天性瓣膜病',
        '诊断分类_心律失常',  '诊断分类_心肌病', '诊断分类_心脏肿瘤',
        '诊断分类_感染性心脏瓣膜病', '诊断分类_缺血性心脏病', '诊断分类_缺血性瓣膜病',
        '诊断分类_退行性心脏瓣膜病', '诊断分类_风湿性心脏瓣膜病', '入院诊断_2型糖尿病',
        '入院诊断_二尖瓣脱垂伴关闭不全', '入院诊断_冠状动脉粥样硬化性心脏病',
        '入院诊断_升主动脉扩张', '入院诊断_非风湿性三尖瓣关闭不全',
        '入院诊断_非风湿性主动脉瓣关闭不全', '入院诊断_非风湿性二尖瓣关闭不全',
        '入院诊断_高血压病3级（极高危）']
    other_detail = {
        '主动脉弓手术': cols_diag,
        '主动脉窦部手术': cols_diag,
        '冠脉搭桥手术': cols_diag,
        '升主动脉手术': cols_diag,
        '左心耳手术': cols_diag,
        '常规三尖瓣成形术': cols_diag,
        '常规三尖瓣置换术': cols_diag,
        '常规主动脉瓣成形术': cols_diag,
        '常规主动脉瓣置换术': cols_diag,
        '常规二尖瓣成形术': cols_diag,
        '常规二尖瓣置换术': cols_diag,
        '常规肺动脉瓣成形术': cols_diag,
        '常规肺动脉瓣置换术': cols_diag,
        '房颤迷宫手术': cols_diag,
        '经导管三尖瓣置换术': cols_diag,
        '经导管主动脉瓣置换术': cols_diag,
        '经导管二尖瓣成形术': cols_diag,
        '经导管二尖瓣置换术': cols_diag,
        '经导管肺动脉瓣置换术': cols_diag,
        '肺动脉导管检查记录': cols_diag,
        '腹主动脉手术': cols_diag,
        '降主动脉手术': cols_diag
    }
    constraint_matrix = set_constraint_matrix(
        data_module, cols_independent, cols_dependent, ind_dep_supp, other_detail)

    # 模型训练
    pl.seed_everything(seed=123)

    lightning_module = DECIModule(
        noise_dist=ContinuousNoiseDist.GAUSSIAN,
        init_rho=30.0,
        init_alpha=0.20,
    )
    lightning_module.constraint_matrix = torch.tensor(constraint_matrix)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=2000,
        fast_dev_run=test_run,
        callbacks=[TQDMProgressBar(refresh_rate=19)],
        enable_checkpointing=False,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value'
    )

    trainer.fit(lightning_module, datamodule=data_module)

    torch.save(
        lightning_module.sem_module,
        "/Users/gxz/Desktop/PT/因果发现/result/deci_gaussian.pt")