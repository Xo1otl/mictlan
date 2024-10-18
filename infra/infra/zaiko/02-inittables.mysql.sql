CREATE DATABASE IF NOT EXISTS zaiko;

USE zaiko;

CREATE TABLE if not exists stocks
(
    sub             SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    amount          INT NOT NULL
);

CREATE TABLE if not exists sales
(
    sub             SERIAL PRIMARY KEY,
    price           VARCHAR(255) NOT NULL
);
