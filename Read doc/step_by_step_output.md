# Phase 1
```cmd
(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/data_pipeline.py
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\policies.json  (100 records)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\claims.json  (200 records)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\endorsements.json  (50 records)       
  fraud flags: 42/200 claims (21.0%)
```
```cmd
(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/data_pipeline.py
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\policies.json  (100 records)
```
```cmd
(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/data_pipeline.py
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)
C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>c:/Users/PANKAJ/OneDrive/Desktop/EXL_graphRAG/.venv/Scripts/activate.bat

(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/data_pipeline.py
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)

(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/data_pipeline.py
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)
```
```cmd
(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/data_pipeline.py
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)
Generating dataset: 100 policies, 200 claims, 50 endorsements (seed=42, fraud_rate=0.1)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\policies.json  (100 records)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\claims.json  (200 records)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\endorsements.json  (50 records)       
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\claims.json  (200 records)
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\endorsements.json  (50 records)       
  wrote C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\samples\endorsements.json  (50 records)       
  fraud flags: 42/200 claims (21.0%)
  wrote 100 PDFs -> C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\pdfs
```
# Phase 2
```cmd
(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>scripts/seed_graph.py --reset --apply-schema
  wrote 100 PDFs -> C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG\data\pdfs

```
```cmd
(.venv) C:\Users\PANKAJ\OneDrive\Desktop\EXL_graphRAG>python scripts/seed_graph.py --reset --apply-schema
connected to bolt://localhost:7687
  applying 21 statements from schema.cypher ...
  schema (constraints + indexes) applied
  graph cleared (--reset)
  nodes queued: 790 (Claim=200, Coverage=161, Endorsement=100, FraudFlag=42, Investigator=87, Policy=100, Policyholder=100)
  relationships queued: 690
  claims: 200 | policies: 100 | fraud flags: 42
  graph nodes: Claim=200, Coverage=161, Endorsement=50, FraudFlag=42, Investigator=6, Policy=100, Policyholder=100
  graph relationships: COVERS=161, ENDORSED_BY=50, FRAUD_DETECTED=42, HAS_CLAIM=200, HAS_POLICY=100, INVESTIGATES_CLAIM=87
```