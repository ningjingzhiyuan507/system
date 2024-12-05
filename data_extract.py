"""
仅从目录的json中抽取字段信息
"""
import os
import pandas as pd
import json
import collections
import logging
from data_explore import VAR_LIST_DICT, VAR_LIST


class FieldProcessor:
    """处理字段来源拆分和范围提取的辅助类"""

    @staticmethod
    def key_split(field):
        """
        拆分字段来源，处理入院录相关字段。
        Args:
            field (str): 待处理的字段来源
        Returns:
            tuple 或 str: 拆分后的字段路径
        """
        parts = field.split('.')
        if parts[0] == '入院录':
            parts = ['.'.join(parts[:2])] + parts[2:]
        return tuple(parts) if len(parts) > 1 else parts[0]

    @staticmethod
    def extract_range(file_path):
        """
        从Excel中提取字段范围。
        Args:
            file_path (str): Excel文件路径
        Returns:
            pd.DataFrame: 包含字段范围信息的数据框
        """
        df = pd.read_excel(
            file_path, sheet_name='overall').dropna(subset=['是否纳入'])
        df['字段'] = df['字段'].apply(FieldProcessor.key_split)
        return df


class JsonExtractor:
    """处理JSON文件抽取的主类"""

    def __init__(self, dir_jsons, fields_range):
        """
        初始化JsonExtractor实例。
        Args:
            dir_jsons (str): JSON文件所在目录
            fields_range (pd.DataFrame): 字段范围信息
        """
        self.dir_jsons = dir_jsons
        self.fields_range = fields_range
        self.results = {'patients': collections.defaultdict(list)}

    def extract_all(self):
        """遍历目录中的JSON文件，提取指定字段信息"""
        files = os.listdir(self.dir_jsons)
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            logging.info(f"Processing file: {base_name}")
            file_path = os.path.join(self.dir_jsons, file_name)
            with open(file_path, 'rb') as file:
                json_data = json.load(file)
            self._process_single_json(json_data, base_name)
        self.results['patients'] = pd.DataFrame(self.results['patients'])
        return self.results

    def _process_single_json(self, json_data, json_name):
        """提取单个JSON文件的字段信息"""
        self.results['patients']['json_name'].append(json_name)
        for _, row in self.fields_range.iterrows():
            field_key = row['字段']
            values = self._extract_value(field_key, json_data)

            if (field_key in VAR_LIST_DICT or field_key in VAR_LIST) and values is None:
                continue
            if field_key in VAR_LIST_DICT:
                self._process_nested_field(field_key, json_name, values)
            elif field_key in VAR_LIST:
                if field_key not in self.results:
                    self.results[field_key] = []
                self.results[field_key].append({"json_name": json_name, 'value': ','.join(values)})
            else:
                self.results['patients'][row['标准字段名']].append(values)

    def _process_nested_field(self, field_key, json_name, values):
        """处理嵌套结构字段的提取"""
        values_df = pd.DataFrame(values)
        values_df['json_name'] = json_name
        self.results[field_key] = pd.concat([self.results[field_key], values_df]) if field_key in self.results else values_df

    @staticmethod
    def _extract_value(field_path, data):
        """递归提取JSON中指定路径的值"""
        if data is None:
            return None
        if isinstance(field_path, str) or len(field_path) == 1:
            key = field_path if isinstance(field_path, str) else field_path[0]
            value = data.get(key, None)
            if value is None:
                return None
            return value.split(';') if key == '入院诊断' else value
        return JsonExtractor._extract_value(field_path[1:], data.get(field_path[0], {}))

    @staticmethod
    def save_to_csv(results, output_dir):
        """保存结果到CSV文件"""
        for key, content in results.items():
            file_name = '.'.join(key) if isinstance(key, tuple) else key
            output_path = os.path.join(output_dir, f'{file_name}.csv')
            content = pd.DataFrame(content) if isinstance(content, list) else content
            content.to_csv(output_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    # 配置路径
    dir_fields_range = '/Users/gxz/Desktop/PT/因果发现/data/字段抽取-20241202.xlsx'
    dir_jsons = '/Users/gxz/pt/complex/data/demo'
    dir_save = '/Users/gxz/Desktop/PT/因果发现/data/raw'

    # 处理字段范围
    fields_range = FieldProcessor.extract_range(dir_fields_range)

    # 初始化并提取JSON数据
    extractor = JsonExtractor(dir_jsons, fields_range)
    extracted_data = extractor.extract_all()

    # 保存结果到CSV
    JsonExtractor.save_to_csv(extracted_data, dir_save)
    logging.info(f"==== All tasks completed successfully ====")