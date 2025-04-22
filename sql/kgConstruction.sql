CREATE TABLE kg_construction
(
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id           BIGINT,
    dataset_id        VARCHAR(255),
    construction_mode VARCHAR(255),
    config            JSON,
    result            JSON,
    graph_id          VARCHAR(255),
    record_type       VARCHAR(255),
    create_time       DATETIME
);