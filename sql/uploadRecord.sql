CREATE TABLE `upload_record`
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(255) NOT NULL,  -- 新增 task_id 字段，用于标识任务
    raw_path VARCHAR(255),
    clean_path VARCHAR(255),
    schema_registry_path VARCHAR(255),
    report_dir VARCHAR(255),
    output_dir VARCHAR(255),
    process_time DATETIME,
    status VARCHAR(50) DEFAULT 'WAITING'  -- 新增 status 字段，默认状态为 'WAITING'
);