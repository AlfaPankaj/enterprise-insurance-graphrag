-- Option A — first-boot initialization (runs once on an empty volume).
-- Everything the GraphRAG system needs inside a SINGLE PostgreSQL instance:
--   * pgvector  → semantic vector search (VECTOR_BACKEND=pgvector)
--   * AGE       → openCypher property graph (GRAPH_BACKEND=age)
--   * graphrag  → the graph the neo4j-compatible shim talks to
CREATE EXTENSION IF NOT EXISTS vector;

LOAD 'age';
SELECT ag_catalog.create_graph('graphrag');
