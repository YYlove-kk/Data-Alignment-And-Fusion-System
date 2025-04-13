CREATE TABLE `alignment_config`
(
    `id`                   bigint      NOT NULL AUTO_INCREMENT,
    `user_id`              varchar(64) NOT NULL COMMENT '用户ID',
    `time_series_algo`     varchar(50) DEFAULT 'DTW' COMMENT '时序对齐算法',
    `time_granularity`     varchar(50) DEFAULT '秒级' COMMENT '时间粒度',
    `semantic_model`       varchar(50) DEFAULT 'BERT' COMMENT '语义模型',
    `similarity_threshold` double      DEFAULT 0.8 COMMENT '相似度阈值',
    `create_time`          datetime    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
);