-- PostgreSQL dialect sample — SERIAL, JSONB, arrays, ENUM type, plpgsql.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy');

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    profile JSONB DEFAULT '{}'::jsonb,
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE FUNCTION greet(name TEXT) RETURNS TEXT
LANGUAGE sql AS $$ SELECT 'Hello, ' || name $$;

CREATE INDEX idx_users_email ON users (email);
