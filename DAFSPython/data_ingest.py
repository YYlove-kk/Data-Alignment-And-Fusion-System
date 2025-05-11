# File: preprocess_table.py
import argparse
import os
import json
import uuid
import pandas as pd
import great_expectations as ge
from util.schema import map_columns
from util.cleaners import (
    fix_numeric,
    fix_text,
    fix_datetime,
    fill_missing,
)
from great_expectations.execution_engine.pandas_execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator


def preprocess_table(file_path: str, registry_path: str, report_dir: str, clean_dir: str) -> str:
    """
    读取单个表格文件，完成标准化清洗，输出 parquet 路径，并保存验证报告。
    :param file_path: 原始 Excel / CSV 路径
    :param registry_path: schema_registry.json
    :param report_dir: 校验报告保存路径
    :param clean_dir: 输出路径
    :return: 清洗后 parquet 路径
    """
    # 1) 读取文件
    df_raw = pd.read_excel(file_path) if file_path.endswith(".xlsx") \
        else pd.read_csv(file_path)
    # 打印原始数据的列名
    print("原始数据列名:", df_raw.columns)  # 打印原始数据列名

    # 2) 字段映射
    df_raw.columns = map_columns(df_raw.columns, registry_path)
    # 打印映射后的列名
    print("映射后数据列名:", df_raw.columns)  # 打印映射后列名
    # 检查缺失列并添加
    required_columns = ["patient_id", "age", "gender","visit_time","name","admission_time", "discharge_time","chief_complaint","history","disease_name"]  # 列出必需的列
    for col in required_columns:
        if col not in df_raw.columns:
            df_raw[col] = 'Unknown'

    # 3) great‑expectations 校验
    # 创建上下文
    context = ge.get_context()
    execution_engine = PandasExecutionEngine()

    # 创建Batch
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df_raw})


    # 创建 Validator
    ge_df = Validator(execution_engine=execution_engine, batches=[batch])
    if "age" in df_raw.columns:
        df_raw["age"] = pd.to_numeric(df_raw["age"], errors='coerce')
        ge_df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
    for gender_col in ["gender", "性别", "GENDER_NAME"]:
        if gender_col in df_raw.columns:
            ge_df.expect_column_values_to_be_in_set(gender_col, ["男", "女", "男性", "女性"])
    validation = ge_df.validate()

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{uuid.uuid4().hex[:8]}_validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation.to_json_dict(), f, ensure_ascii=False, indent=2)

    # 4) 修正 + 缺失填补
    for col in df_raw.columns:
        if df_raw[col].dtype == "object":
            df_raw[col] = fix_text(df_raw[col])
        elif pd.api.types.is_numeric_dtype(df_raw[col]):
            df_raw[col] = fix_numeric(df_raw[col])
        elif "time" in col.lower() or "date" in col.lower():
            df_raw[col] = fix_datetime(df_raw[col])
        df_raw[col] = fill_missing(df_raw[col])

    # 5) 输出 parquet
    out_path = os.path.join(clean_dir, os.path.basename(file_path).rsplit(".", 1)[0] + ".parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_raw.to_parquet(out_path, index=False)
    return out_path


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", required=True, help="输入原始 Excel 或 CSV 文件路径")
    # parser.add_argument("--registry_path", required=True, help="字段映射规则 schema_registry.json")
    # parser.add_argument("--report_dir", default="./report", help="验证报告输出路径")
    # parser.add_argument("--clean_dir", default="./clean", help="清洗结果输出路径")
    #
    # args = parser.parse_args()
    #
    # clean_path = preprocess_table(args.file_path, args.registry_path, args.report_dir, args.clean_dir)

    file_path = "../data/upload/source/test例.xlsx"
    registry_path = "../schema_registry.json"
    report_dir = "../data/upload/report"
    clean_dir = "../data/upload/clean"

    clean_path = preprocess_table(file_path, registry_path, report_dir, clean_dir)

    print(clean_path)  # Java 侧从 stdout 读取


if __name__ == "__main__":
    main()
