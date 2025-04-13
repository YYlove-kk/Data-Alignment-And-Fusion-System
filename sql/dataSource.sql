CREATE TABLE `data_source`
(
    `id`            bigint       NOT NULL AUTO_INCREMENT,
    `name`          varchar(255) NOT NULL COMMENT '数据源名称',
    `modality_type` varchar(50)  DEFAULT NULL COMMENT '模态类型',
    `institution`   varchar(255) DEFAULT NULL COMMENT '机构名称',
    `description`   text COMMENT '数据源描述',
    `creator`       varchar(100) DEFAULT NULL COMMENT '创建人',
    `create_time`   datetime     DEFAULT CURRENT_TIMESTAMP,
    `update_time`   datetime     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
) COMMENT ='数据源管理表';