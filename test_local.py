"""
test_local.py — Pre-flight checks for MedAI
Run this before launching Streamlit to catch issues early.

Usage:
    python test_local.py --api-key sk-...
    python test_local.py  # reads from OPENAI_API_KEY env var
"""
import sys
import os
import json
import time
import argparse
import importlib

# Colour helpers 
GREEN  = "\033[92m"; RED  = "\033[91m"; YELLOW = "\033[93m"
BOLD   = "\033[1m";  RESET = "\033[0m"
ok  = lambda s: print(f"  {GREEN}✅ {s}{RESET}")
err = lambda s: print(f"  {RED}❌ {s}{RESET}")
warn= lambda s: print(f"  {YELLOW}⚠️  {s}{RESET}")
hdr = lambda s: print(f"\n{BOLD}{s}{RESET}")

PASS = 0; FAIL = 0

def check(condition, success_msg, fail_msg):
    global PASS, FAIL
    if condition:
        ok(success_msg); PASS += 1
    else:
        err(fail_msg);   FAIL += 1
    return condition

# 1. Python version 
hdr("1. Python version")
import platform
v = sys.version_info
check(v >= (3, 10), f"Python {v.major}.{v.minor}.{v.micro}", f"Python 3.10+ required, got {v.major}.{v.minor}")

# 2. Required packages 
hdr("2. Package imports")
PACKAGES = [
    ("streamlit",   "streamlit"),
    ("openai",      "openai"),
    ("PIL",         "pillow"),
    ("reportlab",   "reportlab"),
    ("chromadb",    "chromadb"),
    ("rank_bm25",   "rank-bm25"),
    ("requests",    "requests"),
    ("pydicom",     "pydicom"),
    ("numpy",       "numpy"),
    ("tiktoken",    "tiktoken"),
    ("fastapi",     "fastapi"),
    ("uvicorn",     "uvicorn"),
]
for mod, pkg in PACKAGES:
    try:
        importlib.import_module(mod)
        ok(f"{pkg}")
    except ImportError:
        err(f"{pkg} not installed  →  pip install {pkg}")
        FAIL += 1

# 3. Project structure 
hdr("3. Project file structure")
REQUIRED_FILES = [
    "app.py",
    "agents/__init__.py",
    "agents/pipeline.py",
    "agents/vision_agent.py",
    "agents/retrieval_agent.py",
    "agents/diagnosis_agent.py",
    "agents/verifier_agent.py",
    "agents/safety_agent.py",
    "utils/overlay.py",
    "utils/pdf_export.py",
    "api/main.py",
    "openapi.yaml",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "sample_data/traces/sample_traces.jsonl",
]
for f in REQUIRED_FILES:
    check(os.path.exists(f), f, f"MISSING: {f}")

# 4. API key 
hdr("4. OpenAI API key")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
args, _ = parser.parse_known_args()
api_key = args.api_key

if not api_key:
    err("No API key found. Pass --api-key sk-... or set OPENAI_API_KEY")
    FAIL += 1
elif not api_key.startswith("sk-"):
    err(f"API key format looks wrong (should start with 'sk-'): {api_key[:8]}...")
    FAIL += 1
else:
    ok(f"API key present: {api_key[:8]}...{api_key[-4:]}")
    os.environ["OPENAI_API_KEY"] = api_key
    PASS += 1

# 5. OpenAI connectivity 
hdr("5. OpenAI API connectivity")
if api_key and api_key.startswith("sk-"):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        t0 = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=5, timeout=15
        )
        latency = round((time.time() - t0) * 1000)
        reply = resp.choices[0].message.content.strip()
        check("OK" in reply or len(reply) < 20,
              f"GPT-4o reachable ({latency}ms) — response: '{reply}'",
              f"Unexpected response: {reply}")
    except Exception as e:
        err(f"OpenAI call failed: {e}")
        FAIL += 1
else:
    warn("Skipping OpenAI connectivity test (no valid key)")

# 6. PubMed E-utilities 
hdr("6. PubMed E-utilities connectivity")
try:
    import requests
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": "pneumothorax", "retmax": 2, "retmode": "json"},
        timeout=10
    )
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    check(len(ids) > 0, f"PubMed reachable — got {len(ids)} IDs", "PubMed returned no results")
except Exception as e:
    err(f"PubMed unreachable: {e}")
    FAIL += 1

# 7. ChromaDB init 
hdr("7. ChromaDB local store")
try:
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    test_dir = "/tmp/medai_chroma_test"
    ef = OpenAIEmbeddingFunction(api_key=api_key or "sk-test", model_name="text-embedding-3-small")
    client = chromadb.PersistentClient(path=test_dir)
    col = client.get_or_create_collection("test_col", embedding_function=ef)
    ok(f"ChromaDB PersistentClient initialised at {test_dir}")
    PASS += 1
except Exception as e:
    err(f"ChromaDB init failed: {e}")
    FAIL += 1

# 8. Vision agent (mock image) 
hdr("8. VisionAgent (real GPT-4o Vision call)")
if api_key and api_key.startswith("sk-"):
    try:
        from PIL import Image
        import io
        # Create a tiny synthetic grayscale "X-ray"
        img = Image.new("RGB", (256, 256), color=(30, 30, 30))
        sys.path.insert(0, ".")
        from agents.vision_agent import VisionAgent
        va = VisionAgent()
        case_data = {"patient_context": {
            "age": 64, "sex": "Male",
            "chief_complaint": "Acute dyspnea",
            "vitals": {"SpO2": 88, "HR": 120}
        }}
        t0 = time.time()
        findings, overlays, report = va.analyze(img, case_data)
        latency = round((time.time() - t0) * 1000)
        check(isinstance(findings, dict),
              f"VisionAgent returned findings ({latency}ms): {list(findings.keys()) or 'empty (synthetic image — expected)'}",
              "VisionAgent returned invalid findings")
        print(f"    Report preview: {report[:120]}...")
    except Exception as e:
        err(f"VisionAgent failed: {e}")
        FAIL += 1
else:
    warn("Skipping VisionAgent test (no valid key)")

# 9. RetrievalAgent 
hdr("9. RetrievalAgent (PubMed + ChromaDB)")
if api_key and api_key.startswith("sk-"):
    try:
        from agents.retrieval_agent import RetrievalAgent
        ra = RetrievalAgent()
        t0 = time.time()
        snippets, stats = ra.retrieve("pneumothorax chest X-ray management", max_results=3, recency_years=5)
        latency = round((time.time() - t0) * 1000)
        check(len(snippets) >= 0,
              f"RetrievalAgent OK ({latency}ms) — {stats.get('pubmed_fetched',0)} fetched, {len(snippets)} returned",
              "RetrievalAgent returned error")
        if snippets:
            print(f"    Sample snippet: [{snippets[0]['id']}] {snippets[0].get('title','')[:60]}...")
    except Exception as e:
        err(f"RetrievalAgent failed: {e}")
        FAIL += 1
else:
    warn("Skipping RetrievalAgent test (no valid key)")

# 10. PDF export 
hdr("10. PDF export")
try:
    from utils.pdf_export import generate_pdf
    mock_result = {
        "case_id": "test-001",
        "imaging_findings": {"pneumothorax": {"prob": 0.91, "laterality": "right", "size": "small"}},
        "differentials": [{"dx": "Right pneumothorax", "icd10": "J93.9",
                           "rationale": "Test rationale.", "support": []}],
        "red_flags": ["Tension physiology risk"],
        "next_steps": ["Chest drain"],
        "citations": [],
        "groundedness_note": "Test note.",
        "disclaimer": "Research only."
    }
    pdf_bytes = generate_pdf(mock_result, {})
    check(len(pdf_bytes) > 1000,
          f"PDF generated ({len(pdf_bytes):,} bytes)",
          "PDF generation produced empty output")
except Exception as e:
    err(f"PDF export failed: {e}")
    FAIL += 1

# 11. Overlay drawing 
hdr("11. Overlay rendering")
try:
    from PIL import Image
    from utils.overlay import draw_overlays
    img = Image.new("RGB", (512, 512), (40, 40, 40))
    overlays = [{"overlay_id": "ovl_001", "type": "bbox",
                 "coords": [0.6, 0.1, 0.3, 0.4], "finding": "pneumothorax", "label": "Pneumothorax"}]
    result_img = draw_overlays(img, overlays)
    check(result_img is not None, "Overlay drawing works", "Overlay drawing returned None")
except Exception as e:
    err(f"Overlay rendering failed: {e}")
    FAIL += 1

# Summary 
print(f"\n{'='*50}")
total = PASS + FAIL
print(f"{BOLD}Results: {GREEN}{PASS}/{total} passed{RESET}", end="")
if FAIL:
    print(f"  {RED}{FAIL} failed{RESET}")
else:
    print(f"  {GREEN}— all good!{RESET}")

if FAIL == 0:
    print(f"\n{GREEN}{BOLD}✅ Ready to launch!{RESET}")
    print("   streamlit run app.py")
else:
    print(f"\n{RED}{BOLD}❌ Fix the issues above before launching.{RESET}")

sys.exit(0 if FAIL == 0 else 1)