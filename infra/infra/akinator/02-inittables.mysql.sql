CREATE DATABASE IF NOT EXISTS akinator_db;

USE akinator_db;

CREATE TABLE if not exists categories (
    category_id VARCHAR(36) PRIMARY KEY,
    category_name VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE if not exists cases (
    case_id VARCHAR(36) PRIMARY KEY,
    category_id VARCHAR(36) NOT NULL,
    case_name VARCHAR(255) NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id),
    UNIQUE (category_id, case_name)
);

CREATE TABLE if not exists questions (
    question_id VARCHAR(36) PRIMARY KEY,
    category_id VARCHAR(36) NOT NULL,
    question_text VARCHAR(255) NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id),
    UNIQUE (category_id, question_text)
);

CREATE TABLE if not exists choices (
    choice_id INT AUTO_INCREMENT PRIMARY KEY,
    category_id VARCHAR(36) NOT NULL,
    choice_name VARCHAR(255) NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id),
    UNIQUE (category_id, choice_name)
);

CREATE TABLE if not exists case_question_choices (
    case_question_choice_id VARCHAR(36) PRIMARY KEY,
    case_id VARCHAR(36) NOT NULL,
    question_id VARCHAR(36) NOT NULL,
    choice_id INT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id),
    FOREIGN KEY (question_id) REFERENCES questions (question_id),
    FOREIGN KEY (choice_id) REFERENCES choices (choice_id)
);

CREATE OR REPLACE VIEW p_case AS
SELECT cat.category_name, c.case_name, COUNT(c.case_id) / (
        SELECT COUNT(*)
        FROM cases c2
        WHERE
            c2.category_id = c.category_id
    ) AS p_case
FROM categories cat
    JOIN cases c ON cat.category_id = c.category_id
GROUP BY
    cat.category_name,
    c.case_id,
    c.category_id;

CREATE OR REPLACE VIEW p_choice_given_case_question AS
SELECT cat.category_name, c.case_name, q.question_text, ch.choice_name, COUNT(cqc.choice_id) / (
        SELECT COUNT(cqc2.choice_id)
        FROM case_question_choices cqc2
        WHERE
            cqc2.case_id = c.case_id
            AND cqc2.question_id = q.question_id
    ) AS probability
FROM
    categories cat
    JOIN cases c ON cat.category_id = c.category_id
    JOIN case_question_choices cqc ON c.case_id = cqc.case_id
    JOIN questions q ON cqc.question_id = q.question_id
    JOIN choices ch ON cqc.choice_id = ch.choice_id
GROUP BY
    cat.category_name,
    c.case_id,
    q.question_id,
    ch.choice_id;

CREATE OR REPLACE VIEW choices_with_category_name AS
SELECT cat.category_name, ch.choice_name
FROM choices ch
    JOIN categories cat ON ch.category_id = cat.category_id;