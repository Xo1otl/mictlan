-- Active: 1734432854840@@mysql@3306@koemade
CREATE DATABASE IF NOT EXISTS koemade;

USE koemade;
SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

CREATE TABLE IF NOT EXISTS accounts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    status ENUM('active', 'banned', 'suspended') NOT NULL DEFAULT 'active',
    INDEX idx_username (username)
);

CREATE TABLE IF NOT EXISTS roles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS account_role (
    account_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (account_id, role_id),
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS actor_ranks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS actor_profiles (
    display_name VARCHAR(255) NOT NULL,
    rank_id BIGINT NOT NULL,
    self_promotion TEXT,
    price INT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT '受付中',
    account_id BIGINT PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    FOREIGN KEY (rank_id) REFERENCES actor_ranks (id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS nsfw_options (
    allowed BOOLEAN NOT NULL,
    price INT NOT NULL,
    extreme_allowed BOOLEAN NOT NULL,
    extreme_surcharge INT NOT NULL,
    account_id BIGINT PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS profile_images (
    account_id BIGINT PRIMARY KEY NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    size INTEGER NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS voices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    account_id BIGINT NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    path TEXT NOT NULL,
    hash VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    INDEX (account_id),
    UNIQUE (title, account_id)
);

CREATE TABLE IF NOT EXISTS tags (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(255) NOT NULL,
    UNIQUE (name, category)
);

CREATE TABLE IF NOT EXISTS voice_tag (
    voice_id BIGINT NOT NULL,
    tag_id BIGINT NOT NULL,
    PRIMARY KEY (voice_id, tag_id),
    FOREIGN KEY (voice_id) REFERENCES voices (id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
);

-- Insert data into the tags table with the corresponding tag types
INSERT INTO tags (name, category) VALUES ('10代', '年代別タグ');
INSERT INTO tags (name, category) VALUES ('20代', '年代別タグ');
INSERT INTO tags (name, category) VALUES ('30代以上', '年代別タグ');
INSERT INTO tags (name, category) VALUES ('大人しい', 'キャラ別タグ');
INSERT INTO tags (name, category) VALUES ('快活', 'キャラ別タグ');
INSERT INTO tags (name, category) VALUES ('セクシー・渋め', 'キャラ別タグ');

-- Insert admin user
INSERT INTO accounts (username, password) VALUES (
    'admin@koemade.net',
    '$2y$10$1ohq7F7XDTRZa7L9y6FYVui1Bq/8ncdFU0fWeS1ALLBo0z4C4u8qm' -- Abcd1234*
);

-- Insert admin role
INSERT INTO roles (name) VALUES ('admin');

-- Get the admin account_id and role_id
SET @admin_account_id = LAST_INSERT_ID();
SET @admin_role_id = (SELECT id FROM roles WHERE name = 'admin');

-- Insert role into account_role table
INSERT INTO account_role (account_id, role_id) VALUES (@admin_account_id, @admin_role_id);

-- Insert another user
INSERT INTO accounts (username, password) VALUES (
    'qlovolp.ttt@gmail.com',
    '$2y$10$1ohq7F7XDTRZa7L9y6FYVui1Bq/8ncdFU0fWeS1ALLBo0z4C4u8qm' -- Abcd1234*
);

-- Insert actor role
INSERT INTO roles (name) VALUES ('actor');

-- Get the actor account_id and role_id
SET @actor_account_id = LAST_INSERT_ID();
SET @actor_role_id = (SELECT id FROM roles WHERE name = 'actor');

-- Insert role into account_role table
INSERT INTO account_role (account_id, role_id) VALUES (@actor_account_id, @actor_role_id);
