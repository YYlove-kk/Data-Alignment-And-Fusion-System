CREATE TABLE fusion_record
(
    id            INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    graph_tag     INT          NOT NULL,
    UNIQUE (patient_id, graph_tag)
);