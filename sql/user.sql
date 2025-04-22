CREATE TABLE user
(
    id       BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255),
    password VARCHAR(255),
    email    VARCHAR(255),
    addtime  DATETIME
);