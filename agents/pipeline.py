"""
MedAI Agent Pipeline
Agents: Orchestrator → Vision → Retrieval → Diagnosis → CitationVerifier → Safety
"""
import os
import time
import json
import logging
from datetime import datetime
from PIL import Image
from agents.vision_agent    import VisionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.verifier_agent  import CitationVerifierAgent
from agents.safety_agent    import SafetyAgent

logger = logging.getLogger(__name__)

# Singleton agents that can be initialized once, reused across requests 
_vision    = None
_retrieval = None
_diagnosis = None
_verifier  = CitationVerifierAgent()
_safety    = SafetyAgent()

def _get_agents():
    global _vision, _retrieval, _diagnosis
    if _vision    is None: _vision    = VisionAgent()
    if _retrieval is None: _retrieval = RetrievalAgent()
    if _diagnosis is None: _diagnosis = DiagnosisAgent()
    return _vision, _retrieval, _diagnosis, _verifier, _safety

ORCHESTRATOR_PLAN = [
    "Analyze imaging with Vision Agent",
    "Retrieve evidence via PubMed + ChromaDB",
    "Synthesize differential diagnosis",
    "Verify citation groundedness",
    "Run safety & compliance checks"
]


def _retry(fn, retries=2, label=""):
    """Simple retry wrapper for agent calls."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                logger.error(f"{label} failed after {retries+1} attempts: {e}")
                raise
            logger.warning(f"{label} attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(1.5 ** attempt)


def run_pipeline(case_data: dict, image_pil: Image.Image, on_step=None) -> tuple[dict, list]:
    traces = []
    vision, retrieval, diagnosis, verifier, safety = _get_agents()

    def _step(i):
        if on_step:
            on_step(i)

    def trace(agent: str, action: str, input_summary: str, input_payload,
              output, start: float, status: str = "success", error: str = ""):
        traces.append({
            "timestamp":     datetime.utcnow().isoformat(),
            "agent":         agent,
            "action":        action,
            "input_summary": input_summary,
            "input_payload": json.dumps(input_payload, default=str)[:400] if input_payload else "",
            "output_preview": json.dumps(output, default=str)[:400] if output else "",
            "duration_ms":   round((time.time() - start) * 1000, 1),
            "status":        status,
            "error":         error
        })

    # STEP 1: Vision Agent 
    _step(0)
    t = time.time()
    try:
        imaging_findings, overlays, vision_report = _retry(
            lambda: vision.analyze(image_pil, case_data), label="VisionAgent"
        )
        trace("VisionAgent", "analyze_cxr",
              f"Image size: {image_pil.size}",
              {"modality": "CXR", "clinical_context": case_data["patient_context"].get("chief_complaint")},
              imaging_findings, t)
    except Exception as e:
        trace("VisionAgent", "analyze_cxr", "image", None, None, t, "error", str(e))
        imaging_findings, overlays, vision_report = {}, [], f"Vision error: {e}"

    # STEP 2: Retrieval Agent 
    _step(1)
    t = time.time()
    query = _build_retrieval_query(case_data, imaging_findings)
    try:
        snippets, rag_stats = _retry(
            lambda: retrieval.retrieve(
                query=query,
                max_results=case_data["preferences"].get("max_citations", 8),
                recency_years=case_data["preferences"].get("recency_years", 5)
            ), label="RetrievalAgent"
        )
        trace("RetrievalAgent", "hybrid_search",
              f"Query: {query[:120]}",
              {"query": query, "max_results": case_data["preferences"].get("max_citations", 8)},
              f"{len(snippets)} snippets retrieved", t)
    except Exception as e:
        trace("RetrievalAgent", "hybrid_search", query, None, None, t, "error", str(e))
        snippets, rag_stats = [], {}

    # STEP 3: Diagnosis Agent 
    _step(2)
    t = time.time()
    try:
        report = _retry(
            lambda: diagnosis.reason(
                case_data=case_data,
                imaging_findings=imaging_findings,
                vision_report=vision_report,
                snippets=snippets
            ), label="DiagnosisAgent"
        )
        trace("DiagnosisAgent", "generate_report",
              f"Findings: {list(imaging_findings.keys())}, Snippets: {len(snippets)}",
              {"findings": imaging_findings, "snippet_count": len(snippets)},
              f"{len(report.get('differentials',[]))} differentials", t)
    except Exception as e:
        trace("DiagnosisAgent", "generate_report", "", None, None, t, "error", str(e))
        report = {"differentials": [], "red_flags": [], "next_steps": [], "citations": []}

    # STEP 4: Citation Verifier 
    _step(3)
    t = time.time()
    try:
        report, groundedness_note = verifier.verify(report, snippets)
        trace("CitationVerifierAgent", "verify_citations",
              f"Checking {len(report.get('citations',[]))} citations",
              {"citation_ids": [c.get("id") for c in report.get("citations",[])]},
              groundedness_note, t)
    except Exception as e:
        trace("CitationVerifierAgent", "verify_citations", "", None, None, t, "error", str(e))
        groundedness_note = "Verification skipped due to error."

    # STEP 5: Safety Agent 
    _step(4)
    t = time.time()
    try:
        report = safety.check(report)
        trace("SafetyAgent", "compliance_check",
              "PHI check, dosing guardrails, disclaimers",
              {"checks": ["phi", "dosing", "disclaimer"]},
              report.get("disclaimer",""), t)
    except Exception as e:
        trace("SafetyAgent", "compliance_check", "", None, None, t, "error", str(e))
        report["disclaimer"] = "Research/education only. Not for clinical use."

    # Orchestrator trace (prepended) 
    ctx = case_data.get("patient_context", {})
    total_ms = sum(tr.get("duration_ms", 0) for tr in traces)
    traces.insert(0, {
        "timestamp":      datetime.utcnow().isoformat(),
        "agent":          "OrchestratorAgent",
        "action":         "plan_and_route",
        "plan":           ORCHESTRATOR_PLAN,
        "input_summary":  f"Case: {case_data.get('case_id','?')}, Patient: {ctx.get('age','?')}y {ctx.get('sex','?')}",
        "input_payload":  json.dumps(ctx, default=str)[:400],
        "output_preview": f"Pipeline complete. {len(report.get('differentials',[]))} differentials, {len(report.get('citations',[]))} citations.",
        "duration_ms":    round(total_ms, 1),
        "status":         "success",
        "error":          ""
    })

    # Persist traces to sample_data/traces/ 
    traces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "traces")
    os.makedirs(traces_dir, exist_ok=True)
    with open(os.path.join(traces_dir, "run.jsonl"), "a") as f:
        for tr in traces:
            f.write(json.dumps(tr, default=str) + "\n")

    result = {
        "case_id":           case_data.get("case_id", "unknown"),
        "imaging_findings":  imaging_findings,
        "overlays":          overlays,
        "differentials":     report.get("differentials", []),
        "red_flags":         report.get("red_flags", []),
        "next_steps":        report.get("next_steps", []),
        "citations":         report.get("citations", []),
        "groundedness_note": groundedness_note,
        "disclaimer":        report.get("disclaimer", "Research/education only. Not for clinical use."),
        "rag_stats":         rag_stats,
    }
    return result, traces


def _build_retrieval_query(case_data: dict, imaging_findings: dict) -> str:
    ctx     = case_data["patient_context"]
    cc      = ctx.get("chief_complaint", "")
    vitals  = ctx.get("vitals", {})
    findings_str = ", ".join(imaging_findings.keys()) if imaging_findings else "no specific findings"
    query = (
        f"{cc}. Chest X-ray findings: {findings_str}. "
        f"Clinical context: SpO2 {vitals.get('SpO2','?')}%, HR {vitals.get('HR','?')} bpm, "
        f"RR {vitals.get('RR','?')}. "
        f"Differential diagnosis and evidence-based management guidelines."
    )
    return query.strip()