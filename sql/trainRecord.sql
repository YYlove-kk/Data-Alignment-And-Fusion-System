CREATE TABLE train_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,         -- 自增的主键
    epoch INT NOT NULL,                           -- 训练的 epoch
    status VARCHAR(20) NOT NULL,                  -- 训练状态
    resultPath VARCHAR(255),                      -- 模型权重保存路径
    log TEXT,                                     -- 错误信息或日志
    createTime DATETIME NOT NULL,                 -- 创建时间
    hits1 VARCHAR(20),                            -- Hits@1
    hits5 VARCHAR(20),                            -- Hits@5
    hits10 VARCHAR(20),                           -- Hits@10
    trainLoss VARCHAR(20),                        -- 训练损失
    testLoss VARCHAR(20)                          -- 测试损失
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
