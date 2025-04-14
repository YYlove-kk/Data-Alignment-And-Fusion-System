CREATE TABLE kg_node_detail
(
    id          BIGINT PRIMARY KEY COMMENT '主键',
    node_id     VARCHAR(64) NOT NULL COMMENT '节点唯一ID',
    graph_id    VARCHAR(64) NOT NULL COMMENT '所属图谱ID',
    node_type   VARCHAR(32) NOT NULL COMMENT '节点类型',
    properties  JSON COMMENT '节点属性（JSON格式）',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_graph_id (graph_id),
    INDEX idx_node_type (node_type),
    UNIQUE KEY uk_node_graph (node_id, graph_id)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4 COMMENT ='知识图谱节点详情表';