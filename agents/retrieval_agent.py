"""
Retrieval Agent — Hybrid BM25 + vector search over local ChromaDB index.
Fetches fresh abstracts from PubMed E-utilities at runtime.
"""
import os
import re
import time
import json
import hashlib
import requests
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rank_bm25 import BM25Okapi


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EMAIL = "medai-research@example.com"  # required by NCBI policy


class RetrievalAgent:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self._init_chroma()

    def _init_chroma(self):
        persist_dir = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma_store")
        os.makedirs(persist_dir, exist_ok=True)
        ef = OpenAIEmbeddingFunction(api_key=self.api_key, model_name="text-embedding-3-small")
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="medai_literature",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

    # ── PubMed E-utilities ───────────────────────────────────────────────────
    def _pubmed_search(self, query: str, max_results: int, recency_years: int) -> list[dict]:
        from datetime import datetime
        mindate = datetime.now().year - recency_years
        params = {
            "db": "pubmed", "term": query,
            "retmax": max_results, "sort": "relevance",
            "mindate": str(mindate), "maxdate": "3000",
            "datetype": "pdat", "retmode": "json",
            "email": PUBMED_EMAIL
        }
        try:
            r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=10)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            return self._pubmed_fetch(ids)
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []

    def _pubmed_fetch(self, pmids: list[str]) -> list[dict]:
        params = {
            "db": "pubmed", "id": ",".join(pmids),
            "rettype": "abstract", "retmode": "xml",
            "email": PUBMED_EMAIL
        }
        try:
            r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=15)
            r.raise_for_status()
            return self._parse_pubmed_xml(r.text)
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []

    def _parse_pubmed_xml(self, xml_text: str) -> list[dict]:
        import xml.etree.ElementTree as ET
        docs = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid  = article.findtext(".//PMID", "")
                    title = article.findtext(".//ArticleTitle", "")
                    abs_texts = article.findall(".//AbstractText")
                    abstract = " ".join(a.text or "" for a in abs_texts if a.text)
                    year  = article.findtext(".//PubDate/Year", "")
                    doi_el = article.find(".//ELocationID[@EIdType='doi']")
                    doi   = doi_el.text if doi_el is not None else ""
                    # Study type heuristic
                    pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
                    study_type = _classify_study_type(pub_types, title)
                    if abstract:
                        docs.append({
                            "pmid": pmid, "title": title,
                            "abstract": abstract, "year": year,
                            "doi": doi, "study_type": study_type
                        })
                except Exception:
                    continue
        except Exception as e:
            print(f"XML parse error: {e}")
        return docs

    # ── ChromaDB upsert ──────────────────────────────────────────────────────
    def _upsert_docs(self, docs: list[dict]):
        if not docs:
            return
        ids, texts, metas = [], [], []
        for d in docs:
            doc_id = f"pmid_{d['pmid']}" if d.get("pmid") else hashlib.md5(d["title"].encode()).hexdigest()
            text   = f"{d['title']} {d['abstract']}"
            ids.append(doc_id)
            texts.append(text)
            metas.append({k: v for k, v in d.items() if k != "abstract"})
        # Upsert in batches of 50
        for i in range(0, len(ids), 50):
            self.collection.upsert(ids=ids[i:i+50], documents=texts[i:i+50], metadatas=metas[i:i+50])

    # ── Hybrid retrieval ─────────────────────────────────────────────────────
    def retrieve(self, query: str, max_results: int = 8, recency_years: int = 5) -> tuple[list, dict]:
        t0 = time.time()

        # 1. Fetch fresh docs from PubMed and upsert
        fresh_docs = self._pubmed_search(query, max_results * 2, recency_years)
        self._upsert_docs(fresh_docs)

        # 2. Vector search in ChromaDB
        n_results = min(max_results * 2, max(1, self.collection.count()))
        vector_results = self.collection.query(
            query_texts=[query], n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if vector_results and vector_results["documents"]:
            for doc, meta, dist in zip(
                vector_results["documents"][0],
                vector_results["metadatas"][0],
                vector_results["distances"][0]
            ):
                candidates.append({"text": doc, "meta": meta, "vector_score": 1 - dist})

        # 3. BM25 re-rank
        if candidates:
            tokenized = [c["text"].lower().split() for c in candidates]
            bm25 = BM25Okapi(tokenized)
            bm25_scores = bm25.get_scores(query.lower().split())
            for i, c in enumerate(candidates):
                c["bm25_score"] = float(bm25_scores[i])
            max_v = max(c["vector_score"] for c in candidates) or 1
            max_b = max(c["bm25_score"]  for c in candidates) or 1
            for c in candidates:
                c["hybrid_score"] = 0.7 * (c["vector_score"] / max_v) + 0.3 * (c["bm25_score"] / max_b)
            candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 4. Format snippets
        snippets = []
        seen_pmids = set()
        for i, c in enumerate(candidates[:max_results]):
            meta = c.get("meta", {})
            pmid = meta.get("pmid","")
            if pmid and pmid in seen_pmids:
                continue
            if pmid:
                seen_pmids.add(pmid)
            # Extract best sentence as quote
            quote = _extract_best_sentence(c["text"], query)
            snippets.append({
                "id": f"s{i+1}",
                "pmid": pmid,
                "doi": meta.get("doi",""),
                "title": meta.get("title",""),
                "year": meta.get("year",""),
                "study_type": meta.get("study_type","Literature"),
                "quote": quote,
                "full_text": c["text"][:500],
                "hybrid_score": round(c.get("hybrid_score", 0), 4)
            })

        rag_stats = {
            "pubmed_fetched": len(fresh_docs),
            "chroma_count": self.collection.count(),
            "candidates": len(candidates),
            "returned": len(snippets),
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }
        return snippets, rag_stats


def _classify_study_type(pub_types: list[str], title: str) -> str:
    types_str = " ".join(pub_types).lower() + " " + title.lower()
    if "systematic review" in types_str or "meta-analysis" in types_str:
        return "Systematic Review/Meta-Analysis"
    if "randomized" in types_str or "rct" in types_str:
        return "RCT"
    if "guideline" in types_str or "practice guideline" in types_str:
        return "Guideline"
    if "review" in types_str:
        return "Review"
    if "case report" in types_str:
        return "Case Report"
    return "Observational Study"


def _extract_best_sentence(text: str, query: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    query_words = set(query.lower().split())
    best, best_score = sentences[0] if sentences else text[:150], 0
    for s in sentences:
        score = sum(1 for w in s.lower().split() if w in query_words)
        if score > best_score and len(s) > 30:
            best_score, best = score, s
    return best[:250]
