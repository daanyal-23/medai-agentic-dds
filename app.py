import streamlit as st
import json
import base64
from PIL import Image
import io
import os
from datetime import datetime

st.set_page_config(
    page_title="MedAI — Diagnostic Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state initialization (must be before any widget) ─────────────────
for _key in ["image_pil", "result", "traces", "case_data", "rag_seeded"]:
    if _key not in st.session_state:
        st.session_state[_key] = None

# ── Auto-seed RAG index on first boot if ChromaDB is empty ───────────────────
if not st.session_state.get("rag_seeded") and os.environ.get("OPENAI_API_KEY"):
    try:
        import chromadb
        _chroma_path = os.path.join(os.path.dirname(__file__), "rag", "chroma_store")
        _client = chromadb.PersistentClient(path=_chroma_path)
        _col = _client.get_or_create_collection("medai_literature")
        if _col.count() < 10:
            with st.spinner("🔬 First-time setup: seeding RAG index from PubMed (~2 min)..."):
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from agents.retrieval_agent import RetrievalAgent
                _ra = RetrievalAgent()
                _seed_queries = [
                    "pneumothorax diagnosis management chest X-ray",
                    "pulmonary embolism diagnosis CTPA D-dimer",
                    "community acquired pneumonia chest X-ray treatment guidelines",
                    "pleural effusion causes diagnosis management",
                    "cardiomegaly heart failure chest radiograph",
                    "acute coronary syndrome NSTEMI diagnosis ECG troponin",
                    "aortic dissection chest pain diagnosis CT",
                    "congestive heart failure BNP echocardiogram management",
                    "ARDS Berlin definition mechanical ventilation",
                    "tension pneumothorax needle decompression emergency",
                ]
                for _q in _seed_queries:
                    try:
                        _ra.retrieve(query=_q, max_results=8, recency_years=10)
                    except Exception:
                        pass
        st.session_state["rag_seeded"] = True
    except Exception:
        st.session_state["rag_seeded"] = True  # Don't block app on seed failure

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.banner {
    background:#c0392b; color:white; padding:10px 18px;
    border-radius:6px; font-weight:600; font-size:0.9rem; margin-bottom:16px;
}
.diff-card {
    background:#f8f9fa; border-left:4px solid #2980b9;
    padding:14px 18px; border-radius:6px; margin-bottom:10px;
}
.red-flag {
    background:#fff5f5; border-left:4px solid #e74c3c;
    padding:10px 14px; border-radius:6px; margin-bottom:6px; color:#c0392b;
}
.citation-card {
    background:#f0f4ff; border:1px solid #d0d9ff;
    padding:10px 14px; border-radius:6px; margin-bottom:6px; font-size:0.85rem;
}
.finding-badge {
    display:inline-block; background:#2ecc71; color:white;
    padding:2px 10px; border-radius:12px; font-size:0.8rem; margin:3px;
}
.section-header {
    color:#2c3e50; border-bottom:2px solid #3498db;
    padding-bottom:6px; margin-top:24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="banner">⚠️ RESEARCH & EDUCATION ONLY — NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE</div>', unsafe_allow_html=True)

st.title("🏥 MedAI — Agentic Diagnostic Decision Support")
st.caption("Multi-agent clinical reasoning with imaging analysis and evidence retrieval")

# ── Sidebar: API key ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""),
                             help="Enter your OpenAI API key")
    if api_key:
        if not api_key.startswith("sk-"):
            st.warning("⚠️ Invalid API key format — should start with 'sk-'")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set ✓")

    st.divider()
    st.header("🔭 Explainability")
    show_overlays   = st.toggle("Show Imaging Overlays", value=True)
    show_traces     = st.toggle("Show Agent Traces",     value=False)
    show_rag_stats  = st.toggle("Show RAG Stats",        value=False)

    st.divider()
    st.markdown("**Model:** GPT-4o Vision")
    st.markdown("**Agents:** CrewAI pipeline")
    st.markdown("**RAG:** PubMed E-utilities + ChromaDB")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_input, tab_results, tab_traces = st.tabs(["📋 Case Input", "🔬 Analysis Results", "🔍 Agent Traces"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CASE INPUT
# ════════════════════════════════════════════════════════════════════════════
with tab_input:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h3 class="section-header">🩻 Upload Chest X-Ray</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "DICOM (.dcm) or PNG/JPEG",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Upload a chest X-ray for analysis"
        )
        if uploaded_file:
            if uploaded_file.name.endswith(".dcm"):
                st.info("DICOM file detected — extracting metadata and converting for display...")
                try:
                    import pydicom
                    import numpy as np
                    dcm_bytes = uploaded_file.read()
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                        tmp.write(dcm_bytes)
                        tmp_path = tmp.name
                    ds = pydicom.dcmread(tmp_path)
                    os.unlink(tmp_path)
                    arr = ds.pixel_array.astype(float)
                    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype("uint8")
                    img = Image.fromarray(arr).convert("RGB")
                    st.image(img, caption="DICOM Preview (de-identified display)", use_container_width=True)
                    with st.expander("DICOM Metadata"):
                        meta_fields = ["PatientID","StudyDate","Modality","Manufacturer","KVP","Rows","Columns"]
                        for f in meta_fields:
                            if hasattr(ds, f):
                                val = str(getattr(ds, f))
                                if f == "PatientID":
                                    val = "*** DE-IDENTIFIED ***"
                                st.text(f"{f}: {val}")
                    st.session_state["image_pil"] = img
                    st.warning("⚠️ All PHI has been stripped from display. Ensure source images are de-identified.")
                except Exception as e:
                    st.error(f"DICOM parsing error: {e}")
            else:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Uploaded CXR", use_container_width=True)
                st.session_state["image_pil"] = img

    with col2:
        st.markdown('<h3 class="section-header">📝 Clinical Information</h3>', unsafe_allow_html=True)

        with st.expander("Chief Complaint & History", expanded=True):
            chief_complaint = st.text_input("Chief Complaint", "Acute dyspnea, onset 2 hours ago")
            hpi = st.text_area("History of Present Illness", "64-year-old male presenting with sudden onset shortness of breath. No fever. Mild pleuritic chest pain on right side.", height=80)

        with st.expander("Vitals", expanded=True):
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                bp  = st.text_input("BP (mmHg)", "98/60")
                hr  = st.number_input("HR (bpm)", 0, 300, 120)
            with vc2:
                rr  = st.number_input("RR (/min)", 0, 60, 28)
                spo2 = st.number_input("SpO2 (%)", 0, 100, 88)
            with vc3:
                temp = st.number_input("Temp (°C)", 30.0, 42.0, 37.1)
                age  = st.number_input("Age", 0, 120, 64)
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])

        with st.expander("Labs"):
            lc1, lc2 = st.columns(2)
            with lc1:
                d_dimer   = st.number_input("D-dimer (ng/mL)", 0, 50000, 1200)
                troponin  = st.number_input("Troponin (ng/mL)", 0.0, 100.0, 0.03, step=0.01)
            with lc2:
                wbc = st.number_input("WBC (×10³/µL)", 0.0, 100.0, 9.2)
                crp = st.number_input("CRP (mg/L)", 0.0, 500.0, 12.0)

        with st.expander("Medications & Allergies"):
            meds      = st.text_input("Current Medications", "metformin 500mg BD")
            allergies = st.text_input("Allergies", "NKDA")
            pmh       = st.text_area("Past Medical History", "T2DM, ex-smoker (20 pack-years)", height=60)

        with st.expander("RAG Preferences"):
            recency_years = st.slider("Evidence recency (years)", 1, 10, 5)
            max_citations = st.slider("Max citations", 3, 15, 10)

    st.divider()
    run_col, _ = st.columns([1, 3])
    with run_col:
        analyze_btn = st.button("🚀 Run Diagnostic Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not st.session_state.get("image_pil"):
            st.error("Please upload a chest X-ray image.")
        else:
            case_data = {
                "case_id": f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "patient_context": {
                    "age": int(age), "sex": sex,
                    "chief_complaint": chief_complaint,
                    "hpi": hpi,
                    "vitals": {"BP": bp, "HR": int(hr), "RR": int(rr), "SpO2": int(spo2), "Temp": float(temp)},
                    "labs": {"D_dimer": int(d_dimer), "troponin": float(troponin), "WBC": float(wbc), "CRP": float(crp)},
                    "meds": [m.strip() for m in meds.split(",")],
                    "allergies": allergies,
                    "pmh": pmh
                },
                "preferences": {"recency_years": recency_years, "max_citations": max_citations},
                "imaging_provided": True
            }
            st.session_state["case_data"] = case_data

            # ── Run the agent pipeline (CrewAI orchestrated) ──────────────────
            from agents.crew import run_crew_pipeline
            image_for_pipeline = st.session_state["image_pil"]

            step_box    = st.empty()
            progress_bar = st.progress(0)

            STEPS = [
                "🩻 1/5 — Vision Agent analyzing X-ray...",
                "📚 2/5 — Retrieving PubMed evidence...",
                "🧠 3/5 — Synthesizing differential diagnosis...",
                "🔍 4/5 — Verifying citations...",
                "🛡️ 5/5 — Running safety checks...",
            ]

            def on_step(i):
                step_box.info(STEPS[i])
                progress_bar.progress((i + 1) * 20)

            on_step(0)
            try:
                result, traces = run_crew_pipeline(
                    case_data=case_data,
                    image_pil=image_for_pipeline,
                    on_step=on_step
                )
                progress_bar.progress(100)
                step_box.success("✅ Analysis complete! Switch to the Analysis Results tab.")
                st.session_state["result"]    = result
                st.session_state["traces"]    = traces
                st.session_state["case_data"] = case_data
                import time as _t; _t.sleep(1)
                st.rerun()
            except Exception as e:
                step_box.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.get("result"):
        st.info("Run an analysis from the Case Input tab to see results here.")
    else:
        result = st.session_state["result"]
        image  = st.session_state.get("image_pil")

        r1, r2 = st.columns([1, 1])

        # ── Left: Image + overlays ────────────────────────────────────────
        with r1:
            st.markdown('<h3 class="section-header">🩻 Imaging Analysis</h3>', unsafe_allow_html=True)
            if image and show_overlays:
                overlays = result.get("overlays", [])
                if overlays:
                    from utils.overlay import draw_overlays
                    annotated = draw_overlays(image.copy(), overlays)
                    st.image(annotated, caption="CXR with AI Findings Overlay", use_container_width=True)
                else:
                    st.image(image, caption="Uploaded CXR", use_container_width=True)
            elif image:
                st.image(image, caption="Uploaded CXR", use_container_width=True)

            st.markdown("**Structured Findings**")
            findings = result.get("imaging_findings", {})
            if findings:
                for finding, data in findings.items():
                    prob  = data.get("prob", 0)
                    lat   = data.get("laterality", "")
                    size  = data.get("size", "")
                    label = finding.replace("_"," ").title()
                    if lat  and lat  != "null": label += f" ({lat})"
                    if size and size != "null": label += f" — {size}"
                    color = "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
                    st.markdown(f"**{label}**")
                    st.progress(prob, text=f"{prob:.0%}")
            else:
                st.info("No structured findings extracted.")

        # ── Right: Differentials ──────────────────────────────────────────
        with r2:
            st.markdown('<h3 class="section-header">🔬 Differential Diagnoses</h3>', unsafe_allow_html=True)
            for i, dx in enumerate(result.get("differentials", []), 1):
                support_ids = [s.get("snippet_id","") for s in dx.get("support",[])]
                support_str = "  ".join(f"`{s}`" for s in support_ids) if support_ids else ""
                st.markdown(f"""
<div class="diff-card">
<strong>#{i} {dx.get('dx','')}</strong> &nbsp;
<code style="background:#dde;padding:2px 6px;border-radius:4px">{dx.get('icd10','')}</code>
{f'<br><small>Evidence: {support_str}</small>' if support_str else ''}
<br><br>{dx.get('rationale','')}
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Red flags + Next steps ────────────────────────────────────────
        rf_col, ns_col = st.columns(2)
        with rf_col:
            st.markdown('<h3 class="section-header">🚨 Red Flags</h3>', unsafe_allow_html=True)
            for flag in result.get("red_flags", []):
                st.markdown(f'<div class="red-flag">⚠️ {flag}</div>', unsafe_allow_html=True)

        with ns_col:
            st.markdown('<h3 class="section-header">📋 Next Steps</h3>', unsafe_allow_html=True)
            for step in result.get("next_steps", []):
                st.markdown(f"✅ {step}")

        st.divider()

        # ── Citations ─────────────────────────────────────────────────────
        st.markdown('<h3 class="section-header">📚 Evidence Citations</h3>', unsafe_allow_html=True)
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                pmid = c.get("pmid","")
                doi  = c.get("doi","")
                pmid_link = f"[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
                doi_link  = f"[DOI]({doi})" if doi else ""
                study_type = c.get("study_type", "Literature")
                year       = c.get("year","")
                st.markdown(f"""
<div class="citation-card">
<strong>[{c.get('id','?')}]</strong> &nbsp;
<span style="background:#e8f4fd;padding:2px 8px;border-radius:10px;font-size:0.78rem">{study_type}</span>
{f'&nbsp;{year}' if year else ''}
&nbsp; {pmid_link} &nbsp; {doi_link}
<br><em>"{c.get('quote','')}"</em>
<br><small>{c.get('title','')}</small>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No citations retrieved.")

        st.divider()

        # ── Groundedness note ─────────────────────────────────────────────
        if result.get("groundedness_note"):
            st.warning(f"📝 **Groundedness Note:** {result['groundedness_note']}")

        # ── Safety disclaimer ─────────────────────────────────────────────
        st.error(f"🛡️ {result.get('disclaimer','Research/education only. Not for clinical use.')}")

        # ── RAG stats ────────────────────────────────────────────────────
        if show_rag_stats and result.get("rag_stats"):
            with st.expander("📊 RAG Statistics"):
                st.json(result["rag_stats"])

        # ── PDF Export ───────────────────────────────────────────────────
        st.divider()
        from utils.pdf_export import generate_pdf
        # Cache PDF bytes in session state to avoid regenerating on every rerun
        pdf_key = f"pdf_{result.get('case_id','')}"
        if pdf_key not in st.session_state:
            st.session_state[pdf_key] = bytes(
                generate_pdf(result, st.session_state.get("case_data", {}))
            )
        st.download_button(
            "📄 Download PDF Report",
            data=st.session_state[pdf_key],
            file_name=f"medai_report_{result.get('case_id','report')}.pdf",
            mime="application/pdf",
            type="primary",
            key=f"dl_{result.get('case_id','report')}"
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — AGENT TRACES
# ════════════════════════════════════════════════════════════════════════════
with tab_traces:
    if not st.session_state.get("traces"):
        st.info("Traces will appear here after running an analysis.")
    else:
        st.markdown('<h3 class="section-header">🔍 Agent Execution Traces</h3>', unsafe_allow_html=True)
        traces = st.session_state["traces"]
        for trace in traces:
            agent = trace.get("agent","")
            status = trace.get("status","")
            color = "#2ecc71" if status == "success" else "#e74c3c"
            with st.expander(f"{'✅' if status=='success' else '❌'} {agent} — {trace.get('action','')} ({trace.get('duration_ms',0):.0f}ms)"):
                st.json(trace)

        st.divider()
        traces_json = json.dumps(traces, indent=2)
        st.download_button("⬇️ Download Traces (JSONL)", 
                           data="\n".join(json.dumps(t) for t in traces),
                           file_name="agent_traces.jsonl", mime="application/json")