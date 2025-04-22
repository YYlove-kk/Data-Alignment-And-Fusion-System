CREATE TABLE kg_node
(
    node_id        VARCHAR(255) PRIMARY KEY,
    graph_id       VARCHAR(255),
    node_type      VARCHAR(255),
    name           VARCHAR(255),
    data_source    VARCHAR(255),
    version        INT      DEFAULT 1,
    extended_props JSON,
    create_time    DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);