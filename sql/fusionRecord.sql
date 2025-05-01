CREATE TABLE fusion_record
(
    id            INT AUTO_INCREMENT PRIMARY KEY,
    source_ids VARCHAR(255) NOT NULL,
    graph_tag     INT          NOT NULL,
    UNIQUE (source_ids, graph_tag)
);