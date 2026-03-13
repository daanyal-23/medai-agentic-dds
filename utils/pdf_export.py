"""Generate a PDF report from the analysis result."""
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_pdf(result: dict, case_data: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    elements = []

    # Styles 
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                  fontSize=18, spaceAfter=6, textColor=colors.HexColor("#2c3e50"))
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"],
                                  fontSize=13, textColor=colors.HexColor("#2980b9"), spaceAfter=4)
    body        = styles["Normal"]
    small       = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.grey)
    warn_style  = ParagraphStyle("Warn", parent=body, fontSize=9,
                                  textColor=colors.HexColor("#c0392b"),
                                  backColor=colors.HexColor("#fff5f5"), borderPadding=4)
    quote_style = ParagraphStyle("Quote", parent=body, fontSize=9,
                                  textColor=colors.HexColor("#2c3e50"),
                                  leftIndent=12, fontName="Helvetica-Oblique")

    # Header 
    elements.append(Paragraph("🏥 MedAI — Diagnostic Decision Support Report", title_style))
    elements.append(Paragraph(
        f"Case ID: {result.get('case_id','')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        small))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3498db")))
    elements.append(Spacer(1, 0.3*cm))

    # Disclaimer banner 
    elements.append(Paragraph(
        f"⚠️  {result.get('disclaimer','Research/education only. Not for clinical use.')}",
        warn_style))
    elements.append(Spacer(1, 0.4*cm))

    # Patient context 
    ctx = case_data.get("patient_context", {})
    if ctx:
        elements.append(Paragraph("Patient Context", h2_style))
        vitals = ctx.get("vitals", {})
        pt_data = [
            ["Age / Sex", f"{ctx.get('age','?')}y / {ctx.get('sex','?')}"],
            ["Chief Complaint", ctx.get("chief_complaint","")],
            ["Vitals", f"BP {vitals.get('BP','?')} | HR {vitals.get('HR','?')} | RR {vitals.get('RR','?')} | SpO2 {vitals.get('SpO2','?')}%"],
            ["PMH", ctx.get("pmh","")],
            ["Medications", ", ".join(ctx.get("meds",[]))]
        ]
        t = Table(pt_data, colWidths=[4*cm, 13*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eaf2fb")),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d0d0d0")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.4*cm))

    # Imaging findings 
    findings = result.get("imaging_findings", {})
    if findings:
        elements.append(Paragraph("Imaging Findings", h2_style))
        rows = [["Finding", "Probability", "Laterality", "Size"]]
        for name, data in findings.items():
            prob = data.get("prob", 0)
            rows.append([
                name.replace("_"," ").title(),
                f"{prob:.0%}",
                data.get("laterality","—"),
                data.get("size","—")
            ])
        t = Table(rows, colWidths=[6*cm, 3*cm, 4*cm, 4*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2980b9")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d0d0d0")),
            ("ROWBACKGROUNDS", (1,0), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.4*cm))

    # Differentials 
    differentials = result.get("differentials", [])
    if differentials:
        elements.append(Paragraph("Differential Diagnoses", h2_style))
        for i, dx in enumerate(differentials, 1):
            support_ids = [s.get("snippet_id","") for s in dx.get("support",[])]
            support_str = " ".join(f"[{s}]" for s in support_ids)
            block = [
                Paragraph(f"<b>#{i} {dx.get('dx','')} — {dx.get('icd10','')}</b> {support_str}", body),
                Paragraph(dx.get("rationale",""), body),
            ]
            elements.append(KeepTogether(block))
            elements.append(Spacer(1, 0.2*cm))
        elements.append(Spacer(1, 0.2*cm))

    # Red flags 
    red_flags = result.get("red_flags", [])
    if red_flags:
        elements.append(Paragraph("🚨 Red Flags", h2_style))
        for flag in red_flags:
            elements.append(Paragraph(f"⚠️  {flag}", warn_style))
            elements.append(Spacer(1, 0.1*cm))
        elements.append(Spacer(1, 0.2*cm))

    # Next steps 
    next_steps = result.get("next_steps", [])
    if next_steps:
        elements.append(Paragraph("Recommended Next Steps", h2_style))
        for step in next_steps:
            elements.append(Paragraph(f"• {step}", body))
        elements.append(Spacer(1, 0.4*cm))

    # Citations 
    citations = result.get("citations", [])
    if citations:
        elements.append(Paragraph("Evidence Citations", h2_style))
        for c in citations:
            pmid_text = f"PMID: {c.get('pmid','')}" if c.get("pmid") else ""
            doi_text  = f"DOI: {c.get('doi','')}"    if c.get("doi")  else ""
            ref_line  = f"[{c.get('id','')}] {c.get('title','')} — {c.get('study_type','')} {c.get('year','')} {pmid_text} {doi_text}"
            elements.append(Paragraph(ref_line, small))
            if c.get("quote"):
                elements.append(Paragraph(f'"{c["quote"]}"', quote_style))
            elements.append(Spacer(1, 0.15*cm))

    # Groundedness 
    note = result.get("groundedness_note","")
    if note:
        elements.append(Spacer(1, 0.2*cm))
        elements.append(Paragraph(f"Groundedness Note: {note}", small))

    # Footer 
    elements.append(Spacer(1, 0.5*cm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Paragraph(
        "MedAI v0.1 — Research & Education Only — Not a Medical Device", small))

    doc.build(elements)
    return buf.getvalue()
