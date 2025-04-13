CREATE TABLE `upload_record`
(
    `id`               bigint       NOT NULL AUTO_INCREMENT,
    `data_source_name` varchar(255) DEFAULT NULL,
    `modality_type`    varchar(50)  DEFAULT NULL,
    `institution`      varchar(255) DEFAULT NULL,
    `file_path`        varchar(512) NOT NULL,
    `file_name`        varchar(255) NOT NULL,
    `file_size`        bigint       DEFAULT NULL,
    `status`           varchar(20)  DEFAULT '处理中',
    `error_detail`     text,
    `upload_time`      datetime     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
);