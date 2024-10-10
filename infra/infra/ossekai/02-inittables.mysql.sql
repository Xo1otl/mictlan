CREATE DATABASE IF NOT EXISTS ossekai;

USE ossekai;

CREATE TABLE if not exists demo
(
    id             SERIAL PRIMARY KEY,
    name           VARCHAR(255) NOT NULL
);
