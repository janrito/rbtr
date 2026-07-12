-- Greeter schema — tables, views, and statements for greetings.
--
-- The SQL plugin extracts one chunk per top-level statement: CREATE
-- TABLE/VIEW (as classes), CREATE FUNCTION and DML/CTEs (as functions),
-- and CREATE SCHEMA/INDEX/SEQUENCE (as variables). There is no import
-- concept. CREATE PROCEDURE and PRAGMA do not parse and live in the
-- xfail registry instead.

CREATE SCHEMA greet;

CREATE TABLE greeters (
    id INTEGER PRIMARY KEY,
    prefix TEXT NOT NULL
);

CREATE INDEX idx_greeters_prefix ON greeters (prefix);

CREATE SEQUENCE greeting_id_seq;

CREATE VIEW active_greeters AS
SELECT id, prefix FROM greeters WHERE prefix IS NOT NULL;

CREATE FUNCTION shout(message TEXT) RETURNS TEXT AS 'SELECT upper(message)';

WITH recent AS (
    SELECT id FROM greeters
)

SELECT id FROM recent;

INSERT INTO greeters (id, prefix) VALUES (1, 'Hello');

UPDATE greeters SET prefix = 'Hi' WHERE id = 1;

DELETE FROM greeters WHERE id = 1;
