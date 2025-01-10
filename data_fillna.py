"""
1.缺失值填充；
2.分类变量编码。
"""
import os.path

import pandas as pd


VALUE_REPLACE = {
    '是否正常': {'正常': 0, '异常': 1},
    'CC-TT': {'CT': 0, 'CC': 1, 'TT': 2},
    '*1*1': {
        '*1*1': 0, '*1*17': 1, '*1*2': 2, '*1*3': 3, '*17*17': 4, '*2*17': 5,
        '*2*2': 6, '*2*3': 7, '*3*17': 8, '*3*3': 9
    },
    'GG-AA': {'GG': 0, 'GA': 1, 'AA': 2},
    '阴阳性': {'阴性': 0, '阳性': 1},
    '清浊': {'清': 0, '浊': 1},
    '是否找到': {'未找到': 0, '找到': 1},
    '免疫缺陷': {'(－)阴性': 0, '结果待复检': 1},
    '抗体筛选': {'-': 0, '+': 0},
    '梅毒': {'(－)阴性': 0, '呈阳性反应': 1},
    '性状': {'软': 0, '异常': 1},
    '药物敏感性': {'S': 0, 'I': 1, 'R': 2, 'O': 3, 'QT': 4},
    '血型': {'A': 0, 'O': 1, 'B': 2, 'AB': 3},
    '睡眠质量': {'良好': 0, '欠佳': 1},
    '体重': {'正常': 0, '减少': 1, '增加': 2},
    '性别': {'男': 0, '女': 1},
    '民族': {'汉族': 1, '其他': 0},
    '婚姻': {'已婚': 1, '未婚': 2, '其他': 0, '丧偶': 3, '离婚': 4},
    '心功能分级': {'III': 2, 'II': 1, 'I': 0, 'IV': 3},
    '黏膜': {'红润': 0, '苍白': 1, '黄染': 1},
    '胸部叩诊': {'清音': 0, '异常': 1},
    '营养': {'中等': 0, '欠佳': 1, '过剩': 2},
    '呼吸': {'平稳': 0, '浅快': 1, '深长': 2},
    '体位': {'自如': 0, '被动': 1, '强迫': 1},
    '神志': {'清楚': 0, '欠清楚': 1, '昏迷': 2, '嗜睡': 3},
    '精神': {'良好': 0, '一般': 1, '萎靡': 2},
    '健康状况': {'健康': 0, '死亡': 1, '患病': 2},
    '表情': {'自如': 0, '淡漠': 1, '痛苦': 2},
    'TF': {False: 0, True: 1},
    '籍贯': {'江苏省': 1, '上海市': 2, '其他': 0, '安徽省': 3, '浙江省': 4},
    '费用类型': {'异地参保': 2, '医保': 1, '自费': 3, '异地医保': 2, '城镇职工': 1, '城镇居民': 1, '其他': 0},
    '结局': {'治愈': 1, '好转': 2, '其他': 0, '未愈': 3, '自动出院': 4, '死亡': 5},
}

GREATER_1 = [  # 大于0的即为1
    '会阴冲洗', '作业疗法', '关节松动训练', '动脉采血器', '动静脉置管护理', '医用雾化器',
    '压力传感器', '口腔护理', '呼吸过滤器', '固定带', '外科敷料', '尿袋', '换药', '擦浴',
    '气垫床', '气管插管护理', '经电支镜治疗', '肌肉注射', '负压引流器', '起搏器', '麻醉'
]


class FillNaReplace:
    def __init__(self, path_df, path_config):
        self.df = (
            pd.read_csv(os.path.join(path_df, 'integration.csv'))
            .sort_values(by=['json_name', '时间'])
        )
        self.config = pd.read_excel(
            os.path.join(path_config, '字段缺失情况.xlsx'),
            usecols=['字段', '缺失处理方法', '数据类型', '值替换'])

    def main_process(self, path_save):
        for _, row in self.config.iterrows():
            field, fillna, dtype, replace = row
            self.fillna(field, fillna)
            self.replace_value(field, replace)
            self.col_astype(field, dtype)

        self.df.to_csv(os.path.join(path_save, 'integration_fillna.csv'), index=False)

    def fillna(self, field, fillna):
        """插补缺失值"""
        if fillna == '临近值':
            self.df[field] = self.df.groupby('json_name')[field].apply(
                lambda group: group.fillna(method='ffill').fillna(
                    method='bfill')
            ).reset_index(drop=True)
        elif fillna == '删除':
            self.df = self.df.drop(columns=field)
        elif fillna not in ['不处理', '临近值', '删除']:
            self.df[field] = self.df[field].fillna(value=fillna)

    def replace_value(self, field, replace):
        """分类变量值编码"""
        if pd.notna(replace):
            self.df[field] = self.df[field].replace(VALUE_REPLACE[replace])
        if field in GREATER_1:
            self.df.loc[self.df[field] > 0, field] = 1

    def col_astype(self, field, dtype):
        """变量类型转换"""
        if dtype == '连续':
            self.df[field] = pd.to_numeric(self.df[field], errors='coerce')
        elif dtype == '分类':
            self.df[field] = self.df[field].astype('Int64')


if __name__ == '__main__':
    path_df = '/Users/gxz/Desktop/PT/因果发现/data/process/'
    path_config = '/Users/gxz/Desktop/PT/因果发现/data/'
    path_save = '/Users/gxz/Desktop/PT/因果发现/data/process/'

    processer = FillNaReplace(path_df, path_config)
    processer.main_process(path_save)



