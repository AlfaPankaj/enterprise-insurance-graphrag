// ============================================================================
// GraphRAG Insurance Claims System — Neo4j Schema
// Phase 1 deliverable: constraints + indexes (Neo4j 5.x syntax)
// Apply with:  `cypher-shell -f docs/schema.cypher`   (or via neo4j browser)
// All statements are idempotent (IF NOT EXISTS) and safe to re-run.
// ============================================================================

// ---------------------------------------------------------------------------
// 1. UNIQUENESS CONSTRAINTS — one per entity label (no duplicate entities)
// ---------------------------------------------------------------------------
CREATE CONSTRAINT policyholder_id_unique  IF NOT EXISTS FOR (n:Policyholder) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT policy_id_unique        IF NOT EXISTS FOR (n:Policy)       REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT coverage_id_unique      IF NOT EXISTS FOR (n:Coverage)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT claim_id_unique         IF NOT EXISTS FOR (n:Claim)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT endorsement_id_unique   IF NOT EXISTS FOR (n:Endorsement)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT fraudflag_id_unique     IF NOT EXISTS FOR (n:FraudFlag)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT investigator_id_unique  IF NOT EXISTS FOR (n:Investigator) REQUIRE n.id IS UNIQUE;

// ---------------------------------------------------------------------------
// 2. EXISTENCE CONSTRAINTS — REQUIRES Neo4j ENTERPRISE EDITION
// ---------------------------------------------------------------------------
// Node property existence constraints are an Enterprise-only feature and are
// NOT supported on the Community image used for this demo. Uniqueness
// constraints (section 1) + application-level validation (Phase 5 validators)
// cover data quality on Community. Uncomment these if you run Enterprise:
//
// CREATE CONSTRAINT policyholder_id_required   IF NOT EXISTS FOR (n:Policyholder) REQUIRE n.id IS NOT NULL;
// CREATE CONSTRAINT policyholder_name_required IF NOT EXISTS FOR (n:Policyholder) REQUIRE n.name IS NOT NULL;
// CREATE CONSTRAINT policy_id_required         IF NOT EXISTS FOR (n:Policy)       REQUIRE n.id IS NOT NULL;
// CREATE CONSTRAINT policy_number_required     IF NOT EXISTS FOR (n:Policy)       REQUIRE n.policy_number IS NOT NULL;
// CREATE CONSTRAINT policy_type_required       IF NOT EXISTS FOR (n:Policy)       REQUIRE n.type IS NOT NULL;
// CREATE CONSTRAINT claim_id_required          IF NOT EXISTS FOR (n:Claim)        REQUIRE n.id IS NOT NULL;
// CREATE CONSTRAINT claim_number_required      IF NOT EXISTS FOR (n:Claim)        REQUIRE n.claim_number IS NOT NULL;
// CREATE CONSTRAINT claim_amount_required      IF NOT EXISTS FOR (n:Claim)        REQUIRE n.amount IS NOT NULL;
// CREATE CONSTRAINT endorsement_id_required    IF NOT EXISTS FOR (n:Endorsement)  REQUIRE n.id IS NOT NULL;
// CREATE CONSTRAINT fraudflag_id_required      IF NOT EXISTS FOR (n:FraudFlag)    REQUIRE n.id IS NOT NULL;
// CREATE CONSTRAINT fraudflag_reason_required  IF NOT EXISTS FOR (n:FraudFlag)    REQUIRE n.reason IS NOT NULL;
// CREATE CONSTRAINT investigator_id_required   IF NOT EXISTS FOR (n:Investigator) REQUIRE n.id IS NOT NULL;
// CREATE CONSTRAINT investigator_name_required IF NOT EXISTS FOR (n:Investigator) REQUIRE n.name IS NOT NULL;

// ---------------------------------------------------------------------------
// 3. RANGE INDEXES — point lookups & property filters on hot query paths
// ---------------------------------------------------------------------------

// Entity IDs are already indexed by the uniqueness constraints above.
// Index the external-facing numbers too (used in user-facing queries):
CREATE INDEX policy_number_idx IF NOT EXISTS FOR (n:Policy)     ON (n.policy_number);
CREATE INDEX claim_number_idx  IF NOT EXISTS FOR (n:Claim)      ON (n.claim_number);
CREATE INDEX endorsement_number_idx IF NOT EXISTS FOR (n:Endorsement) ON (n.endorsement_number);

// Status filters (dashboard / workflow queries):
CREATE INDEX policy_status_idx IF NOT EXISTS FOR (n:Policy)     ON (n.status);
CREATE INDEX claim_status_idx  IF NOT EXISTS FOR (n:Claim)      ON (n.status);

// Date-range queries (e.g. "claims filed within 7 days of inception"):
CREATE INDEX policy_start_date_idx IF NOT EXISTS FOR (n:Policy) ON (n.start_date);
CREATE INDEX policy_end_date_idx   IF NOT EXISTS FOR (n:Policy) ON (n.end_date);
CREATE INDEX claim_date_idx        IF NOT EXISTS FOR (n:Claim)  ON (n.date);

// Numeric filters (amounts, premiums, risk):
CREATE INDEX claim_amount_idx         IF NOT EXISTS FOR (n:Claim)        ON (n.amount);
CREATE INDEX policy_premium_idx       IF NOT EXISTS FOR (n:Policy)       ON (n.premium);
CREATE INDEX policyholder_risk_idx    IF NOT EXISTS FOR (n:Policyholder) ON (n.risk_score);
CREATE INDEX fraudflag_confidence_idx IF NOT EXISTS FOR (n:FraudFlag)    ON (n.confidence);

// ---------------------------------------------------------------------------
// 4. TEXT INDEXES — fuzzy name search (Phase 3 retrieval + re-ranker)
// ---------------------------------------------------------------------------
CREATE TEXT INDEX policyholder_name_text_idx IF NOT EXISTS FOR (n:Policyholder) ON (n.name);
CREATE TEXT INDEX investigator_name_text_idx  IF NOT EXISTS FOR (n:Investigator) ON (n.name);

// ============================================================================
// Optional: validate the schema
// ============================================================================
// SHOW CONSTRAINTS;
// SHOW INDEXES;
//
// Sanity check that constraints exist:
// SHOW CONSTRAINTS YIELD name RETURN name ORDER BY name;
