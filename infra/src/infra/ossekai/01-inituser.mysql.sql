CREATE USER 'ossekai_user'@'%' IDENTIFIED BY 'ossekai_password';
GRANT ALL PRIVILEGES ON ossekai.* TO 'ossekai_user'@'%';
FLUSH PRIVILEGES;
