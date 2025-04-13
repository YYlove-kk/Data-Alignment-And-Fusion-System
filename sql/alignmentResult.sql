CREATE TABLE `alignment_result`
(
    `id`             bigint NOT NULL AUTO_INCREMENT,
    `config_id`      bigint NOT NULL COMMENT '关联的配置ID',
    `result_json`    longtext COMMENT '对齐结果JSON',
    `chart_data`     text COMMENT '图表数据JSON',
    `avg_similarity` double      DEFAULT NULL COMMENT '平均相似度',
    `status`         varchar(20) DEFAULT '运行中' COMMENT '任务状态',
    `create_time`    datetime    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
);