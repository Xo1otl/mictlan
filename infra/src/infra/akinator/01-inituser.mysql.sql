CREATE USER 'akinator' @'%' IDENTIFIED BY 'akinator_password';

GRANT ALL PRIVILEGES ON akinator_db.* TO 'akinator' @'%';

FLUSH PRIVILEGES;