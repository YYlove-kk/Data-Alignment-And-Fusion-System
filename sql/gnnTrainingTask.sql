CREATE TABLE gnn_training_task
(
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    task_name   VARCHAR(255),
    status      VARCHAR(50),
    result_path TEXT,
    log         TEXT,
    create_time DATETIME,
    update_time DATETIME
);