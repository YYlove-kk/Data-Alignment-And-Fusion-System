CREATE TABLE joint_embedding_relation
(
    id         INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    text_file  VARCHAR(255) NOT NULL,
    image_file VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);