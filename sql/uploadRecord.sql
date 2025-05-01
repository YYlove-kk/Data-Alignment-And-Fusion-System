CREATE TABLE upload_record
(
    id                   INT AUTO_INCREMENT PRIMARY KEY, -- 自增长的主键
    task_id              VARCHAR(255) NOT NULL,          -- 任务ID，唯一标识任务
    source_id            VARCHAR(255),                   -- 来源ID
    file_name            VARCHAR(255),                   -- 文件名
    raw_path             VARCHAR(255),                   -- 原文件路径
    clean_dir            VARCHAR(255),                   -- 清洗结果输出路径
    clean_path           VARCHAR(255),                   -- 清洗路径
    output_dir           VARCHAR(255),                   -- 嵌入结果输出路径
    output_path          VARCHAR(255),                   -- 嵌入结果路径
    schema_registry_path VARCHAR(255),                   -- 模式注册路径
    report_dir           VARCHAR(255),                   -- 清洗报告路径
    modality_type        VARCHAR(255),                   -- 模态类型
    institution          VARCHAR(255),                   -- 机构
    process_time         DATETIME,                       -- 处理时间
    status               VARCHAR(255)                    -- 任务状态
);
