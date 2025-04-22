CREATE TABLE analysis_record
(
    id            BIGINT PRIMARY KEY,
    user_id       BIGINT       NOT NULL,
    analysis_type VARCHAR(255) NOT NULL,
    data_sources  JSON,
    analysis_mode VARCHAR(255) NOT NULL,
    parameters    JSON,
    result_stats  JSON,
    create_time   DATETIME
);