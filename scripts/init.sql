-- FaceCV Database Initialization Script

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS facecv
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE facecv;

-- Create faces table with optimized indexes
CREATE TABLE IF NOT EXISTS faces (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model VARCHAR(50) DEFAULT 'buffalo_l',
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_name (name),
    INDEX idx_created_at (created_at),
    INDEX idx_updated_at (updated_at),
    FULLTEXT idx_metadata (metadata)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create attendance records table
CREATE TABLE IF NOT EXISTS attendance_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    face_id VARCHAR(36),
    name VARCHAR(255) NOT NULL,
    action ENUM('check_in', 'check_out', 'break_in', 'break_out') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT DEFAULT 0.0,
    camera_id VARCHAR(100),
    metadata JSON,
    
    INDEX idx_face_id (face_id),
    INDEX idx_name_date (name, DATE(timestamp)),
    INDEX idx_timestamp (timestamp),
    INDEX idx_camera_id (camera_id),
    
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create stranger alerts table
CREATE TABLE IF NOT EXISTS stranger_alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_level ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    camera_id VARCHAR(100),
    image_path VARCHAR(500),
    confidence FLOAT DEFAULT 0.0,
    appearance_count INT DEFAULT 1,
    metadata JSON,
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_alert_level (alert_level),
    INDEX idx_camera_id (camera_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create video processing logs
CREATE TABLE IF NOT EXISTS video_processing_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    video_source VARCHAR(500) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    frames_processed INT DEFAULT 0,
    faces_detected INT DEFAULT 0,
    status ENUM('processing', 'completed', 'failed') DEFAULT 'processing',
    error_message TEXT,
    metadata JSON,
    
    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create API access logs table
CREATE TABLE IF NOT EXISTS api_access_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT,
    response_time_ms INT,
    client_ip VARCHAR(45),
    user_agent VARCHAR(500),
    api_key VARCHAR(100),
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_endpoint (endpoint),
    INDEX idx_api_key (api_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(20),
    tags JSON,
    
    INDEX idx_timestamp_metric (timestamp, metric_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create database views for reporting
CREATE OR REPLACE VIEW daily_attendance_summary AS
SELECT 
    DATE(timestamp) as date,
    name,
    MIN(CASE WHEN action = 'check_in' THEN timestamp END) as first_check_in,
    MAX(CASE WHEN action = 'check_out' THEN timestamp END) as last_check_out,
    COUNT(CASE WHEN action = 'check_in' THEN 1 END) as check_in_count,
    COUNT(CASE WHEN action = 'check_out' THEN 1 END) as check_out_count
FROM attendance_records
GROUP BY DATE(timestamp), name;

CREATE OR REPLACE VIEW hourly_activity AS
SELECT 
    DATE(timestamp) as date,
    HOUR(timestamp) as hour,
    COUNT(DISTINCT name) as unique_people,
    COUNT(*) as total_detections,
    AVG(confidence) as avg_confidence
FROM attendance_records
GROUP BY DATE(timestamp), HOUR(timestamp);

-- Create stored procedures
DELIMITER //

CREATE PROCEDURE IF NOT EXISTS cleanup_old_logs(IN days_to_keep INT)
BEGIN
    DELETE FROM api_access_logs WHERE timestamp < DATE_SUB(NOW(), INTERVAL days_to_keep DAY);
    DELETE FROM video_processing_logs WHERE start_time < DATE_SUB(NOW(), INTERVAL days_to_keep DAY);
    DELETE FROM performance_metrics WHERE timestamp < DATE_SUB(NOW(), INTERVAL days_to_keep DAY);
END//

CREATE PROCEDURE IF NOT EXISTS get_attendance_report(
    IN start_date DATE,
    IN end_date DATE,
    IN person_name VARCHAR(255)
)
BEGIN
    SELECT 
        DATE(timestamp) as date,
        name,
        MIN(CASE WHEN action = 'check_in' THEN timestamp END) as check_in_time,
        MAX(CASE WHEN action = 'check_out' THEN timestamp END) as check_out_time,
        TIMEDIFF(
            MAX(CASE WHEN action = 'check_out' THEN timestamp END),
            MIN(CASE WHEN action = 'check_in' THEN timestamp END)
        ) as total_time
    FROM attendance_records
    WHERE DATE(timestamp) BETWEEN start_date AND end_date
        AND (person_name IS NULL OR name = person_name)
    GROUP BY DATE(timestamp), name
    ORDER BY date DESC, name;
END//

DELIMITER ;

-- Create initial admin user (optional)
-- INSERT INTO api_users (username, api_key, is_active) VALUES ('admin', UUID(), TRUE);

-- Grant privileges to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON facecv.* TO 'facecv'@'%';
-- FLUSH PRIVILEGES;