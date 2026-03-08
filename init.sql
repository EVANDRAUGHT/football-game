CREATE DATABASE IF NOT EXISTS football_analysis;
USE football_analysis;

CREATE TABLE IF NOT EXISTS videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_uuid VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS posture_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT,
    frame_index INT,
    data JSON,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS athlete_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT,
    score FLOAT,
    analysis_summary TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);
