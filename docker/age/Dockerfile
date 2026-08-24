# Option A — unified PostgreSQL data stack: Apache AGE + pgvector.
#
# Base image ships AGE (openCypher in PostgreSQL); the extra layer installs
# the pgvector extension from the PGDG apt repository so ONE database serves
# relational tables + vector search + the property graph.
#
# Build:  docker compose -f docker-compose.age.yml up -d
# (If the pgvector layer is unavailable in your environment, the plain
#  `apache/age:PG16_latest` image also works — the system falls back to its
#  built-in in-memory vector index; only VECTOR_BACKEND=pgvector needs this.)
FROM apache/age:PG16_latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

USER postgres
