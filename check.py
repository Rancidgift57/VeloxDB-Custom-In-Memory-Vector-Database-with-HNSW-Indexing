from core.database import VectorDatabase

db = VectorDatabase(dim=128, metric="cosine", data_dir="./data")

id_ = db.insert([0.1]*128, metadata={"category": "finance"})
hits = db.search([0.1]*128, k=5, filter={"category": "finance"})
print(hits)

db.close()  # flushes a final snapshot