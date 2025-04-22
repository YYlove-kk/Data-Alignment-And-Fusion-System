CREATE TABLE kg_fusion_record
(
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    main_graph_id    BIGINT       NOT NULL,
    sub_graph_id     BIGINT       NOT NULL,
    strategy_type    VARCHAR(255) NOT NULL,
    conflict_details JSON,
    new_node_count   INT,
    conflict_count   INT,
    create_time      DATETIME
);