CREATE TABLE alignment_detail
(
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    result_id   BIGINT,
    source_data VARCHAR(255),
    target_data VARCHAR(255),
    similarity  DOUBLE
);