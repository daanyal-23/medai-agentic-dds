"""
MedAI FastAPI REST API
POST /analyze-case — matches the assignment's specified API contract
"""
import os
import base64
import io
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.pipeline import run_pipeline

app = FastAPI(
    title="MedAI — Agentic Diagnostic Decision Support",
    description=(
        "Multi-agent clinical reasoning with chest X-ray analysis, "
        "evidence retrieval, and grounded differential diagnosis. "
        "**RESEARCH & EDUCATION ONLY — NOT A MEDICAL DEVICE.**"
    ),
    version="0.1.0",
    contact={"name": "MedAI Research", "email": "medai@example.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class Vitals(BaseModel):
    BP:   str   = "120/80"
    HR:   int   = 80
    RR:   int   = 16
    SpO2: int   = 98
    Temp: float = 37.0

class Labs(BaseModel):
    D_dimer:  Optional[float] = None
    troponin: Optional[float] = None
    WBC:      Optional[float] = None
    CRP:      Optional[float] = None

class PatientContext(BaseModel):
    age:              int
    sex:              str
    chief_complaint:  str
    hpi:              Optional[str] = ""
    vitals:           Vitals        = Field(default_factory=Vitals)
    labs:             Labs          = Field(default_factory=Labs)
    meds:             list[str]     = []
    allergies:        Optional[str] = "NKDA"
    pmh:              Optional[str] = ""

class AnalysisPreferences(BaseModel):
    recency_years: int = 5
    max_citations: int = 8

class AnalyzeCaseRequest(BaseModel):
    case_id:         str
    patient_context: PatientContext
    image_b64:       str = Field(..., description="Base64-encoded PNG/JPEG chest X-ray")
    preferences:     AnalysisPreferences = Field(default_factory=AnalysisPreferences)

    model_config = {"json_schema_extra": {"example": {
        "case_id": "abc-123",
        "patient_context": {
            "age": 64, "sex": "Male",
            "chief_complaint": "Acute dyspnea",
            "vitals": {"BP": "98/60", "HR": 120, "RR": 28, "SpO2": 88},
            "labs": {"D_dimer": 1200, "troponin": 0.03},
            "meds": ["metformin"]
        },
        "image_b64": "<base64-encoded PNG>",
        "preferences": {"recency_years": 5, "max_citations": 8}
    }}}


class ImagingFinding(BaseModel):
    prob:        float
    laterality:  Optional[str] = None
    size:        Optional[str] = None
    description: Optional[str] = None

class Differential(BaseModel):
    dx:        str
    icd10:     str
    rationale: str
    support:   list[dict] = []

class Citation(BaseModel):
    id:         str
    pmid:       Optional[str] = None
    doi:        Optional[str] = None
    title:      Optional[str] = None
    year:       Optional[str] = None
    study_type: Optional[str] = None
    quote:      Optional[str] = None

class Overlay(BaseModel):
    overlay_id: str
    type:       str
    coords:     list[float]
    label:      Optional[str] = None

class RagStats(BaseModel):
    pubmed_fetched: int = 0
    chroma_count:   int = 0
    candidates:     int = 0
    returned:       int = 0
    latency_ms:     float = 0.0

class AnalyzeCaseResponse(BaseModel):
    case_id:           str
    imaging_findings:  dict[str, ImagingFinding]
    differentials:     list[Differential]
    red_flags:         list[str]
    next_steps:        list[str]
    citations:         list[Citation]
    overlays:          list[Overlay]
    groundedness_note: str
    rag_stats:         Optional[RagStats] = None
    disclaimer:        str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "MedAI Diagnostic Decision Support", "version": "0.1.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/analyze-case", response_model=AnalyzeCaseResponse, tags=["Analysis"])
def analyze_case(request: AnalyzeCaseRequest):
    """
    Analyze a chest X-ray + clinical vignette using the multi-agent pipeline.

    Returns differential diagnoses with ICD-10 codes, imaging findings,
    red flags, next steps, and inline citations from PubMed.

    **RESEARCH & EDUCATION ONLY — NOT FOR CLINICAL USE.**
    """
    # Validate API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")

    # Decode image
    try:
        img_bytes = base64.b64decode(request.image_b64)
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

    # Build internal case_data dict
    case_data = {
        "case_id": request.case_id,
        "patient_context": {
            "age":            request.patient_context.age,
            "sex":            request.patient_context.sex,
            "chief_complaint":request.patient_context.chief_complaint,
            "hpi":            request.patient_context.hpi,
            "vitals":         request.patient_context.vitals.model_dump(),
            "labs":           request.patient_context.labs.model_dump(),
            "meds":           request.patient_context.meds,
            "allergies":      request.patient_context.allergies,
            "pmh":            request.patient_context.pmh,
        },
        "preferences": request.preferences.model_dump()
    }

    try:
        result, _ = run_pipeline(case_data=case_data, image_pil=image_pil)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Normalize overlays to ensure overlay_id field
    overlays = []
    for i, ovl in enumerate(result.get("overlays", [])):
        overlays.append({
            "overlay_id": ovl.get("overlay_id", f"ovl_{i+1:03d}"),
            "type":       ovl.get("type", "bbox"),
            "coords":     ovl.get("coords", []),
            "label":      ovl.get("label", ovl.get("finding",""))
        })

    return {
        "case_id":           result["case_id"],
        "imaging_findings":  result.get("imaging_findings", {}),
        "differentials":     result.get("differentials", []),
        "red_flags":         result.get("red_flags", []),
        "next_steps":        result.get("next_steps", []),
        "citations":         result.get("citations", []),
        "overlays":          overlays,
        "groundedness_note": result.get("groundedness_note", ""),
        "rag_stats":         result.get("rag_stats"),
        "disclaimer":        result.get("disclaimer", "Research/education only. Not for clinical use."),
    }
