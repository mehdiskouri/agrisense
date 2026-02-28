-- AgriSense — PostgreSQL initialization script
-- Runs once on first container start via docker-entrypoint-initdb.d

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Verify extensions
DO $$
BEGIN
    RAISE NOTICE 'uuid-ossp version: %', (SELECT extversion FROM pg_extension WHERE extname = 'uuid-ossp');
    RAISE NOTICE 'postgis version: %', (SELECT extversion FROM pg_extension WHERE extname = 'postgis');
END $$;
