"""
seed_index.py — Pre-populate ChromaDB with ~100 PubMed abstracts
for common chest X-ray findings.

Usage:
    OPENAI_API_KEY=sk-... python rag/seed_index.py
"""
import os, sys
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.retrieval_agent import RetrievalAgent

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


SEED_QUERIES = [
    # Original 5 that succeeded
    "pneumothorax diagnosis management chest X-ray",
    "pulmonary embolism diagnosis CTPA D-dimer",
    "community acquired pneumonia chest X-ray treatment guidelines",
    "pleural effusion causes diagnosis management",
    "cardiomegaly heart failure chest radiograph",
    # Replaced with more specific queries to avoid overlap
    "acute pulmonary edema treatment emergency",
    "lobar consolidation pneumonia radiology",
    "tension pneumothorax needle decompression emergency",
    "subsegmental atelectasis postoperative management",
    "ARDS Berlin definition mechanical ventilation",
    # Bonus: extra clinical topics for better citation coverage
    "acute coronary syndrome NSTEMI diagnosis ECG troponin",
    "aortic dissection chest pain diagnosis CT",
    "congestive heart failure BNP echocardiogram management",
    "hypoxemia differential diagnosis SpO2 management",
    "chest X-ray interpretation systematic approach radiology",
]

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    agent = RetrievalAgent()
    total = 0
    for q in SEED_QUERIES:
        print(f"Seeding: {q[:60]}...")
        snippets, stats = agent.retrieve(query=q, max_results=10, recency_years=10)
        total += stats.get("pubmed_fetched", 0)
        print(f"  → {stats['pubmed_fetched']} fetched, {stats['chroma_count']} in store")

    print(f"\n✅ Seed complete. Total fetched: {total}. ChromaDB docs: {agent.collection.count()}")