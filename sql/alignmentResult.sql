CREATE TABLE alignment_result
(
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY, -- 主键
    alignment_matrix    TEXT,                              -- 存储对齐矩阵（使用 JSON 字符串或类似格式）
    semantic_accuracy   DOUBLE,                            -- 存储语义准确率
    alignment_coverage  INT,                               -- 存储对齐覆盖数
    diagonal_similarity TEXT,
    source_ids         TEXT
);
