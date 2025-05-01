CREATE TABLE build_record
(
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    source_ids  VARCHAR(64) NOT NULL,
    graph_tag   INT         NOT NULL,
    mode        INT         NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
