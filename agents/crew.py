"""
agents/crew.py — CrewAI orchestration layer for MedAI
Wraps existing VisionAgent, RetrievalAgent, DiagnosisAgent, etc. as CrewAI Tools.
Defines 5 Agents + 5 Tasks with sequential dependencies.
"""
import os
import json
import base64
import io
import logging
from PIL import Image
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

logger = logging.getLogger(__name__)

# ── Shared state passed between tasks via context ────────────────────────────
class PipelineContext:
    """Mutable context passed through the CrewAI pipeline."""
    def __init__(self, case_data: dict, image_pil: Image.Image):
        self.case_data        = case_data
        self.image_pil        = image_pil
        self.imaging_findings = {}
        self.overlays         = []
        self.vision_report    = ""
        self.snippets         = []
        self.rag_stats        = {}
        self.report           = {}
        self.groundedness_note= ""

# Global context holder (CrewAI tools are stateless, so we use a module-level ref)
_ctx: PipelineContext = None


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — each wraps one of our existing agent classes
# ══════════════════════════════════════════════════════════════════════════════

class VisionAnalysisTool(BaseTool):
    name: str = "vision_analysis"
    description: str = (
        "Analyzes a chest X-ray image using GPT-4o Vision. "
        "Returns structured imaging findings JSON with probabilities, laterality, "
        "size, and bounding box overlays. Input: 'analyze'."
    )

    def _run(self, query: str = "analyze") -> str:
        from agents.vision_agent import VisionAgent
        import time
        global _ctx
        for attempt in range(3):
            try:
                va = VisionAgent()
                findings, overlays, report = va.analyze(_ctx.image_pil, _ctx.case_data)
                if not findings and attempt < 2:
                    logger.warning(f"VisionAgent returned empty findings on attempt {attempt+1}, retrying...")
                    time.sleep(2)
                    continue
                _ctx.imaging_findings = findings
                _ctx.overlays         = overlays
                _ctx.vision_report    = report
                return json.dumps({
                    "findings": findings,
                    "overlays": overlays,
                    "report":   report
                }, default=str)
            except Exception as e:
                logger.error(f"VisionAnalysisTool error (attempt {attempt+1}): {e}")
                if attempt == 2:
                    return json.dumps({"error": str(e), "findings": {}, "overlays": [], "report": ""})
                time.sleep(2)
        return json.dumps({"findings": {}, "overlays": [], "report": "Vision analysis failed after 3 attempts."})


class EvidenceRetrievalTool(BaseTool):
    name: str = "evidence_retrieval"
    description: str = (
        "Retrieves relevant medical literature from PubMed E-utilities and ChromaDB "
        "using hybrid BM25 + vector search. Returns ranked snippets with PMID/DOI. "
        "Input: clinical query string."
    )

    def _run(self, query: str) -> str:
        from agents.retrieval_agent import RetrievalAgent
        from agents.pipeline import _build_retrieval_query
        global _ctx
        try:
            if not query or query.strip() == "retrieve":
                query = _build_retrieval_query(_ctx.case_data, _ctx.imaging_findings)
            ra = RetrievalAgent()
            prefs = _ctx.case_data.get("preferences", {})
            snippets, stats = ra.retrieve(
                query=query,
                max_results=prefs.get("max_citations", 10),
                recency_years=prefs.get("recency_years", 5)
            )
            _ctx.snippets  = snippets
            _ctx.rag_stats = stats
            return json.dumps({
                "snippet_count": len(snippets),
                "snippets":      snippets,
                "rag_stats":     stats
            }, default=str)
        except Exception as e:
            logger.error(f"EvidenceRetrievalTool error: {e}")
            return json.dumps({"error": str(e), "snippets": [], "rag_stats": {}})


class DiagnosisReasoningTool(BaseTool):
    name: str = "diagnosis_reasoning"
    description: str = (
        "Synthesizes imaging findings, clinical context, and evidence snippets to produce "
        "a structured DDS report with differential diagnoses (ICD-10), red flags, "
        "next steps, and inline citations. Input: 'reason'."
    )

    def _run(self, query: str = "reason") -> str:
        from agents.diagnosis_agent import DiagnosisAgent
        global _ctx
        try:
            da = DiagnosisAgent()
            report = da.reason(
                case_data=_ctx.case_data,
                imaging_findings=_ctx.imaging_findings,
                vision_report=_ctx.vision_report,
                snippets=_ctx.snippets
            )
            _ctx.report = report
            return json.dumps(report, default=str)
        except Exception as e:
            logger.error(f"DiagnosisReasoningTool error: {e}")
            return json.dumps({"error": str(e), "differentials": [], "red_flags": [], "next_steps": [], "citations": []})


class CitationVerificationTool(BaseTool):
    name: str = "citation_verification"
    description: str = (
        "Validates that every differential diagnosis has at least one exact quoted snippet "
        "from retrieved literature. Appends a groundedness note. Input: 'verify'."
    )

    def _run(self, query: str = "verify") -> str:
        from agents.verifier_agent import CitationVerifierAgent
        global _ctx
        try:
            cv = CitationVerifierAgent()
            report, note = cv.verify(_ctx.report, _ctx.snippets)
            _ctx.report             = report
            _ctx.groundedness_note  = note
            return json.dumps({"groundedness_note": note, "citation_count": len(report.get("citations", []))})
        except Exception as e:
            logger.error(f"CitationVerificationTool error: {e}")
            return json.dumps({"error": str(e), "groundedness_note": "Verification failed."})


class SafetyComplianceTool(BaseTool):
    name: str = "safety_compliance"
    description: str = (
        "Runs safety and compliance checks: PHI de-identification, dosing guardrails, "
        "and adds mandatory research/education disclaimers. Input: 'check'."
    )

    def _run(self, query: str = "check") -> str:
        from agents.safety_agent import SafetyAgent
        global _ctx
        try:
            sa = SafetyAgent()
            report = sa.check(_ctx.report)
            _ctx.report = report
            return json.dumps({
                "disclaimer":    report.get("disclaimer", ""),
                "safety_flags":  report.get("safety_flags", []),
                "status":        "passed"
            })
        except Exception as e:
            logger.error(f"SafetyComplianceTool error: {e}")
            return json.dumps({"error": str(e), "status": "failed"})


# ══════════════════════════════════════════════════════════════════════════════
# CREWAI AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def build_crew() -> Crew:
    """Build and return the MedAI CrewAI crew."""

    llm_model = "gpt-4o"

    # ── Agent definitions ────────────────────────────────────────────────────

    orchestrator = Agent(
        role="Medical AI Orchestrator",
        goal=(
            "Plan and coordinate the full diagnostic workflow. Route tasks to specialist agents "
            "in order: Vision → Retrieval → Diagnosis → Verification → Safety. "
            "Enforce policy: no dosing without guideline citation, no PHI in output."
        ),
        backstory=(
            "You are the lead coordinator of a medical AI system. You ensure each specialist "
            "agent completes their task before the next begins, and that all outputs meet "
            "clinical safety and compliance standards."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm_model
    )

    vision_agent = Agent(
        role="Radiology Vision Specialist",
        goal=(
            "Analyze chest X-ray images to extract structured findings with probabilities, "
            "laterality, size, and bounding box coordinates for visual overlays."
        ),
        backstory=(
            "You are a specialist in medical image analysis with expertise in chest radiography. "
            "You use GPT-4o Vision to identify pathological findings and produce structured JSON reports."
        ),
        tools=[VisionAnalysisTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm_model
    )

    retrieval_agent = Agent(
        role="Medical Literature Retrieval Specialist",
        goal=(
            "Retrieve the most relevant, recent, high-quality medical evidence from PubMed "
            "and local ChromaDB index using hybrid BM25 + vector search."
        ),
        backstory=(
            "You are a medical librarian AI with access to PubMed E-utilities and a local "
            "vector database. You specialize in finding evidence-based literature to support "
            "clinical reasoning, prioritizing guidelines, systematic reviews, and RCTs."
        ),
        tools=[EvidenceRetrievalTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm_model
    )

    diagnosis_agent = Agent(
        role="Clinical Diagnosis and Reasoning Specialist",
        goal=(
            "Synthesize imaging findings, patient context, and evidence snippets into a "
            "structured diagnostic decision support report with ranked differentials, "
            "ICD-10 codes, red flags, next steps, and inline citations."
        ),
        backstory=(
            "You are a senior clinician AI trained in evidence-based medicine. You integrate "
            "multimodal inputs — imaging, labs, vitals, history, and literature — to produce "
            "grounded differential diagnoses with full citation support."
        ),
        tools=[DiagnosisReasoningTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm_model
    )

    verifier_agent = Agent(
        role="Citation Verification Specialist",
        goal=(
            "Validate that every differential diagnosis has at least one exact verbatim quote "
            "from retrieved literature. Flag any unsupported claims."
        ),
        backstory=(
            "You are a medical fact-checker AI. You ensure all clinical claims in the report "
            "are grounded in retrieved evidence with exact quotes. You flag any differential "
            "that lacks citation support."
        ),
        tools=[CitationVerificationTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm_model
    )

    safety_agent = Agent(
        role="Safety and Compliance Officer",
        goal=(
            "Ensure the report is free of PHI, contains no unsupported dosing recommendations, "
            "and includes mandatory research/education disclaimers."
        ),
        backstory=(
            "You are a medical AI compliance officer. You enforce regulatory guardrails: "
            "no patient identifiers, no dosing without guideline backing, and clear "
            "'Not a medical device' disclaimers on all outputs."
        ),
        tools=[SafetyComplianceTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm_model
    )

    # ── Task definitions ─────────────────────────────────────────────────────

    task_vision = Task(
        description=(
            "Analyze the uploaded chest X-ray image. Use the vision_analysis tool "
            "to extract structured findings (probabilities, laterality, size) and "
            "bounding box overlays. Return findings.json."
        ),
        expected_output=(
            "JSON object with imaging findings (finding name → prob/laterality/size/description), "
            "overlay coordinates, and a 2-3 sentence radiology report summary."
        ),
        agent=vision_agent
    )

    task_retrieval = Task(
        description=(
            "Based on the imaging findings and clinical context, retrieve relevant medical "
            "literature. Use the evidence_retrieval tool with a query combining the chief "
            "complaint, imaging findings, and key vitals. Return ranked snippets with PMID/DOI."
        ),
        expected_output=(
            "List of evidence snippets with PMID, DOI, title, year, study type, and "
            "verbatim quote. At least 5 snippets covering the top differential diagnoses."
        ),
        agent=retrieval_agent,
        context=[task_vision]
    )

    task_diagnosis = Task(
        description=(
            "Using the imaging findings, vision report, patient context, and evidence snippets, "
            "produce a full diagnostic decision support report. Use the diagnosis_reasoning tool. "
            "Include 3-5 ranked differentials with ICD-10 codes, rationale, red flags, "
            "next steps, and inline citations."
        ),
        expected_output=(
            "Structured JSON report with: differentials (dx, icd10, rationale, support), "
            "red_flags, next_steps, and citations (id, pmid, doi, quote)."
        ),
        agent=diagnosis_agent,
        context=[task_vision, task_retrieval]
    )

    task_verification = Task(
        description=(
            "Verify citation groundedness of the diagnostic report. Use the citation_verification "
            "tool to check every differential has at least one exact quoted snippet. "
            "Return a groundedness note."
        ),
        expected_output=(
            "Groundedness note confirming all citations verified, or listing gaps "
            "that require manual review."
        ),
        agent=verifier_agent,
        context=[task_diagnosis]
    )

    task_safety = Task(
        description=(
            "Run safety and compliance checks on the final report. Use the safety_compliance "
            "tool to verify: no PHI present, no unsupported dosing, and research disclaimer added."
        ),
        expected_output=(
            "Compliance confirmation with disclaimer text and list of any safety flags triggered."
        ),
        agent=safety_agent,
        context=[task_verification]
    )

    # ── Assemble crew ────────────────────────────────────────────────────────
    crew = Crew(
        agents=[orchestrator, vision_agent, retrieval_agent, diagnosis_agent, verifier_agent, safety_agent],
        tasks=[task_vision, task_retrieval, task_diagnosis, task_verification, task_safety],
        process=Process.sequential,
        verbose=True
    )

    return crew


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_crew_pipeline(case_data: dict, image_pil: Image.Image, on_step=None) -> tuple[dict, list]:
    """
    Run the MedAI pipeline via CrewAI orchestration.
    Returns (result_dict, traces_list) matching the same interface as run_pipeline().
    """
    import time
    from datetime import datetime

    global _ctx

    # ── Warm up singleton agents before CrewAI starts ────────────────────────
    # This prevents cold-start race conditions on first run after reinitialization.
    from agents.vision_agent    import VisionAgent
    from agents.retrieval_agent import RetrievalAgent
    from agents.diagnosis_agent import DiagnosisAgent
    _warmup_vision    = VisionAgent()
    _warmup_retrieval = RetrievalAgent()
    _warmup_diagnosis = DiagnosisAgent()

    _ctx = PipelineContext(case_data=case_data, image_pil=image_pil)

    traces = []
    t_total = time.time()

    def _step(i):
        if on_step:
            on_step(i)

    MAX_ATTEMPTS = 2  # retry once if result is empty

    for attempt in range(MAX_ATTEMPTS):
        try:
            # Reset context on retry
            _ctx = PipelineContext(case_data=case_data, image_pil=image_pil)

            _step(0)
            crew = build_crew()
            _step(1)
            crew_output = crew.kickoff()
            _step(2); _step(3); _step(4)

            # Check if result is empty — retry if so
            if not _ctx.report.get("differentials") and attempt < MAX_ATTEMPTS - 1:
                logger.warning(f"CrewAI returned empty report on attempt {attempt+1}. Retrying...")
                time.sleep(2)
                continue

            # If still empty after retries, fall back to direct pipeline
            if not _ctx.report.get("differentials"):
                logger.warning("CrewAI returned empty report after all attempts. Falling back to direct pipeline.")
                from agents.pipeline import run_pipeline
                return run_pipeline(case_data=case_data, image_pil=image_pil, on_step=on_step)

            traces = _build_traces(_ctx, time.time() - t_total)

            # Persist traces
            traces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "traces")
            os.makedirs(traces_dir, exist_ok=True)
            with open(os.path.join(traces_dir, "run.jsonl"), "a") as f:
                for tr in traces:
                    f.write(json.dumps(tr, default=str) + "\n")

            result = {
                "case_id":           case_data.get("case_id", "unknown"),
                "imaging_findings":  _ctx.imaging_findings,
                "overlays":          _ctx.overlays,
                "differentials":     _ctx.report.get("differentials", []),
                "red_flags":         _ctx.report.get("red_flags", []),
                "next_steps":        _ctx.report.get("next_steps", []),
                "citations":         _ctx.report.get("citations", []),
                "groundedness_note": _ctx.groundedness_note,
                "disclaimer":        _ctx.report.get("disclaimer", "Research/education only. Not for clinical use."),
                "rag_stats":         _ctx.rag_stats,
            }
            return result, traces

        except Exception as e:
            logger.error(f"CrewAI pipeline error (attempt {attempt+1}): {e}")
            if attempt == MAX_ATTEMPTS - 1:
                logger.warning("Falling back to direct pipeline.")
                from agents.pipeline import run_pipeline
                return run_pipeline(case_data=case_data, image_pil=image_pil, on_step=on_step)
            time.sleep(2)

    # Should never reach here, but safety net
    from agents.pipeline import run_pipeline
    return run_pipeline(case_data=case_data, image_pil=image_pil, on_step=on_step)


def _build_traces(ctx: PipelineContext, total_seconds: float) -> list:
    """Build JSONL-compatible trace records from pipeline context."""
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    return [
        {
            "timestamp": now, "agent": "OrchestratorAgent", "action": "crew_kickoff",
            "framework": "CrewAI", "process": "sequential",
            "input_summary": f"Case: {ctx.case_data.get('case_id','?')}",
            "output_preview": f"{len(ctx.report.get('differentials',[]))} differentials, {len(ctx.report.get('citations',[]))} citations",
            "duration_ms": round(total_seconds * 1000, 1), "status": "success"
        },
        {
            "timestamp": now, "agent": "VisionAgent", "action": "analyze_cxr",
            "framework": "CrewAI/Tool", "tool": "vision_analysis",
            "output_preview": json.dumps(ctx.imaging_findings, default=str)[:300],
            "status": "success" if ctx.imaging_findings else "empty"
        },
        {
            "timestamp": now, "agent": "RetrievalAgent", "action": "hybrid_search",
            "framework": "CrewAI/Tool", "tool": "evidence_retrieval",
            "output_preview": f"{len(ctx.snippets)} snippets, latency {ctx.rag_stats.get('latency_ms','?')}ms",
            "status": "success"
        },
        {
            "timestamp": now, "agent": "DiagnosisAgent", "action": "generate_report",
            "framework": "CrewAI/Tool", "tool": "diagnosis_reasoning",
            "output_preview": f"{len(ctx.report.get('differentials',[]))} differentials",
            "status": "success"
        },
        {
            "timestamp": now, "agent": "CitationVerifierAgent", "action": "verify_citations",
            "framework": "CrewAI/Tool", "tool": "citation_verification",
            "output_preview": ctx.groundedness_note,
            "status": "success"
        },
        {
            "timestamp": now, "agent": "SafetyAgent", "action": "compliance_check",
            "framework": "CrewAI/Tool", "tool": "safety_compliance",
            "output_preview": ctx.report.get("disclaimer", "")[:100],
            "status": "success"
        },
    ]