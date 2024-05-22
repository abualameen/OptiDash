-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS od_dev_db;
-- Create or update the od_dev user with the required privileges
CREATE USER IF NOT EXISTS 'od_dev'@'localhost' IDENTIFIED BY 'od_dev_pwd';
-- Grant all privileges on the od_dev_db database to the od_dev user
GRANT ALL PRIVILEGES ON od_dev_db.* TO 'od_dev'@'localhost';
-- Grant SELECT PRIVILEGE ON THE performance_schema database to the od_dev_db
GRANT SELECT ON performance_schema.* TO 'od_dev'@'localhost';
