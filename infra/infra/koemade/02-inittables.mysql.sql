CREATE DATABASE IF NOT EXISTS koemade;

USE koemade;

CREATE TABLE if not exists signup_requests (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    furigana VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    tel VARCHAR(20) NOT NULL,
    bank_name VARCHAR(255) NOT NULL,
    branch_name VARCHAR(255) NOT NULL,
    account_number VARCHAR(20) NOT NULL,
    self_promotion VARCHAR(500) NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS account_roles (
    account_id BIGINT UNSIGNED NOT NULL,
    role_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (account_id, role_id),
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE
);

CREATE TABLE if not exists profiles (
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(255) NOT NULL,
    self_promotion TEXT,
    price INT NOT NULL,
    account_id BIGINT UNSIGNED PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE if not exists actors_r (
    ok BOOLEAN NOT NULL,
    price INT NOT NULL,
    hard_ok BOOLEAN NOT NULL,
    hard_surcharge INT NOT NULL,
    account_id BIGINT UNSIGNED PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE if not exists profile_images (
    account_id BIGINT UNSIGNED PRIMARY KEY NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    size INTEGER NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE if not exists voices (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    account_id BIGINT UNSIGNED NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    filename VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    INDEX (account_id),
    UNIQUE (title, account_id) -- Ensures title is unique per account_id
);

CREATE TABLE IF NOT EXISTS voice_tags (
    id SERIAL PRIMARY KEY,
    tag_name VARCHAR(255) NOT NULL UNIQUE, -- タグの名前 (例: "10代", "20代", "おとなしい", "快活")
    tag_category VARCHAR(255) NOT NULL --  タグタイプの名前 (例: "年代別タグ", "キャラ別タグ")
);

CREATE TABLE IF NOT EXISTS voice_tag_map (
    voice_id BIGINT UNSIGNED NOT NULL, -- ID of the voice from the voices table
    tag_id BIGINT UNSIGNED NOT NULL, -- ID of the tag from the voice_tags table
    PRIMARY KEY (voice_id, tag_id), -- Composite primary key to ensure unique pairs of voice_id and tag_id
    FOREIGN KEY (voice_id) REFERENCES voices (id) ON DELETE CASCADE, -- Foreign key constraint linking to voices table
    FOREIGN KEY (tag_id) REFERENCES voice_tags (id) ON DELETE CASCADE -- Foreign key constraint linking to voice_tags table
);

-- Abcd1234*
INSERT INTO
    accounts (username, password)
VALUES (
        'admin@koemade.net',
        '$2y$10$1ohq7F7XDTRZa7L9y6FYVui1Bq/8ncdFU0fWeS1ALLBo0z4C4u8qm'
    );

-- Get the last inserted ID
SET @last_id = LAST_INSERT_ID();

-- Insert role into roles table if it does not exist
INSERT INTO
    roles (role_name)
VALUES ('admin')
ON DUPLICATE KEY UPDATE
    id = id;

-- Get the role ID
SET
    @role_id = (
        SELECT id
        FROM roles
        WHERE
            role_name = 'admin'
    );

-- Insert role into account_roles table
INSERT INTO
    account_roles (account_id, role_id)
VALUES (@last_id, @role_id);

-- Insert another user
INSERT INTO
    accounts (username, password)
VALUES (
        'qlovolp.ttt@gmail.com',
        '$2y$10$1ohq7F7XDTRZa7L9y6FYVui1Bq/8ncdFU0fWeS1ALLBo0z4C4u8qm'
    );

SET @last_id = LAST_INSERT_ID();

-- Insert role into roles table if it does not exist
INSERT INTO
    roles (role_name)
VALUES ('actor')
ON DUPLICATE KEY UPDATE
    id = id;

-- Get the role ID
SET
    @role_id = (
        SELECT id
        FROM roles
        WHERE
            role_name = 'actor'
    );

-- Insert role into account_roles table
INSERT INTO
    account_roles (account_id, role_id)
VALUES (@last_id, @role_id);

-- Set the character set and collation for the session
SET NAMES utf8mb4;

SET CHARACTER SET utf8mb4;

-- Insert data into the voice_tags table with the corresponding tag types
INSERT INTO
    voice_tags (tag_name, tag_category)
values ('10代', '年代別タグ');

INSERT INTO
    voice_tags (tag_name, tag_category)
values ('20代', '年代別タグ');

INSERT INTO
    voice_tags (tag_name, tag_category)
values ('30代以上', '年代別タグ');

INSERT INTO
    voice_tags (tag_name, tag_category)
values ('大人しい', 'キャラ別タグ');

INSERT INTO
    voice_tags (tag_name, tag_category)
values ('快活', 'キャラ別タグ');

INSERT INTO
    voice_tags (tag_name, tag_category)
values ('セクシー・渋め', 'キャラ別タグ');