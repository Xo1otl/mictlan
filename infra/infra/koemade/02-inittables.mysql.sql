-- Active: 1737615399853@@mysql@3306@koemade
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

INSERT INTO actor_ranks (name, description) VALUES ('amateur', '初心者向けのランクです');

CREATE TABLE IF NOT EXISTS actor_profiles (
    display_name VARCHAR(255) NOT NULL,
    rank_id BIGINT DEFAULT 1,
    self_promotion TEXT,
    price INT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT '受付中',
    account_id BIGINT PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    FOREIGN KEY (rank_id) REFERENCES actor_ranks (id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS nsfw_options (
    allowed BOOLEAN NOT NULL DEFAULT FALSE,
    price INT NULL,
    extreme_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    extreme_surcharge INT NULL,
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

CREATE OR REPLACE VIEW voices_view AS
SELECT
    v.id AS voice_id,
    v.title AS voice_name,
    ap.account_id AS actor_id,
    ap.display_name AS actor_name,  -- ここをusername → display_nameに変更
    ap.status AS actor_status,
    ar.name AS actor_rank,
    (SELECT COUNT(*) FROM voices vv WHERE vv.account_id = ap.account_id) AS total_voices,
    (
        SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
                'id', t.id,
                'category', t.category,
                'name', t.name
            )
        )
        FROM voice_tag vt
        JOIN tags t ON vt.tag_id = t.id
        WHERE vt.voice_id = v.id
    ) AS tags,
    v.path AS source_url
FROM
    voices v
LEFT JOIN
    actor_profiles ap ON v.account_id = ap.account_id
LEFT JOIN
    actor_ranks ar ON ap.rank_id = ar.id  -- accountsテーブルの結合を削除
GROUP BY
    v.id, ap.account_id, ap.display_name, ar.name, v.path;  -- GROUP BYを調整
    
CREATE OR REPLACE VIEW actor_info_view AS
SELECT
    ap.account_id AS actor_id,
    ap.display_name AS actor_name,
    ap.status AS actor_status,
    ar.name AS actor_rank,
    ap.self_promotion AS actor_description,
    pi.path AS actor_avatar_url,
    ap.price AS actor_price_default,
    nsfw.allowed AS actor_nsfw_allowed,
    nsfw.price AS actor_price_nsfw,
    nsfw.extreme_allowed AS actor_nsfw_extreme_allowed,
    nsfw.extreme_surcharge AS actor_price_nsfw_extreme
FROM
    actor_profiles ap
LEFT JOIN
    actor_ranks ar ON ap.rank_id = ar.id
LEFT JOIN
    profile_images pi ON ap.account_id = pi.account_id
LEFT JOIN
    nsfw_options nsfw ON ap.account_id = nsfw.account_id;
 
CREATE OR REPLACE VIEW actor_feed_view AS
SELECT
    ap.account_id AS actor_id,
    ap.display_name AS actor_name,
    ap.status AS actor_status,
    ar.name AS actor_rank,
    ap.self_promotion AS actor_description,
    pi.path AS actor_avatar_url,
    ap.price AS actor_price_default,
    nsfw.allowed AS actor_nsfw_allowed,
    nsfw.price AS actor_price_nsfw,
    nsfw.extreme_allowed AS actor_nsfw_extreme_allowed,
    nsfw.extreme_surcharge AS actor_price_nsfw_extreme,
    v.id AS voice_id,
    v.title AS voice_title,
    v.path AS voice_source_url,
    t.id AS tag_id,
    t.name AS tag_name,
    t.category AS tag_category
FROM
    actor_profiles ap
LEFT JOIN
    actor_ranks ar ON ap.rank_id = ar.id
LEFT JOIN
    profile_images pi ON ap.account_id = pi.account_id
LEFT JOIN
    nsfw_options nsfw ON ap.account_id = nsfw.account_id
LEFT JOIN
    voices v ON ap.account_id = v.account_id
LEFT JOIN
    voice_tag vt ON v.id = vt.voice_id
LEFT JOIN
    tags t ON vt.tag_id = t.id;   

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
