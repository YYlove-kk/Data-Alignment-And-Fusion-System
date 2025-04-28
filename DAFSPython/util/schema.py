import json

def map_columns(columns, registry_path):
    """
    将原始列名映射为标准名
    :param columns: 原始列名列表
    :param registry_path: schema_registry.json 路径
    :return: 映射后的列名列表
    """
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return [registry.get(col.lower(), col) for col in columns]