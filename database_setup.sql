-- Person profiles table
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    face_encoding BYTEA,
    appearance_features JSONB,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP
);

-- Detection events table
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(id),
    event_type VARCHAR(50),
    confidence FLOAT,
    timestamp TIMESTAMP,
    camera_id VARCHAR(50),
    bbox JSONB
);
