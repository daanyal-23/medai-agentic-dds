"""
Diagnosis/Reasoning Agent — synthesizes imaging findings + clinical context + evidence
to produce a structured DDS report with differentials, ICD-10, red flags, next steps.
"""
import os
import re
import json
from openai import OpenAI


DIAGNOSIS_SYSTEM = """You are a senior clinical AI reasoning assistant (research/education only).
Given structured imaging findings, patient context, and evidence snippets, produce a 
Diagnostic Decision Support report.

You MUST return a valid JSON object with this exact structure:
{
  "differentials": [
    {
      "dx": "Diagnosis name",
      "icd10": "ICD-10 code",
      "rationale": "2-3 sentence clinical rationale citing findings and evidence",
      "support": [{"snippet_id": "s1"}, {"snippet_id": "s2"}]
    }
  ],
  "red_flags": ["list of urgent/emergent considerations as strings"],
  "next_steps": ["recommended next tests and referrals as strings"],
  "citations": [
    {
      "id": "s1",
      "pmid": "from snippet",
      "doi": "from snippet",
      "title": "from snippet",
      "year": "from snippet",
      "study_type": "from snippet",
      "quote": "exact quoted sentence from snippet supporting a claim"
    }
  ]
}

CRITICAL RULES:
- Provide 3-5 ranked differentials, most likely first
- EVERY differential MUST have at least one citation in its support array — use the best matching snippet even if imperfect
- If a snippet partially supports a differential, still cite it and note the partial relevance in rationale
- Citations must use the exact quote from the provided snippets — pick the most relevant sentence
- ICD-10 codes must be valid
- Red flags should be urgent/emergent only
- Do NOT recommend imaging that has already been provided (see Imaging Status in prompt)
- Do NOT recommend dosing unless directly quoted from a guideline snippet
- Return ONLY valid JSON, no markdown fences
"""


class DiagnosisAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model  = "gpt-4o"

    def reason(self, case_data: dict, imaging_findings: dict,
                vision_report: str, snippets: list[dict]) -> dict:
        ctx = case_data["patient_context"]

        # Build prompt
        snippets_text = "\n".join([
            f"[{s['id']}] ({s.get('study_type','')}, {s.get('year','')}): {s.get('quote','')} | Full: {s.get('full_text','')[:200]}"
            for s in snippets
        ])

        clinical_context_note = (
            "NOTE: A chest X-ray image HAS already been uploaded and analyzed by the Vision Agent. "
            "Do NOT recommend obtaining a chest X-ray as a next step — it has already been done. "
            "Base imaging next steps on findings from the vision report above (e.g. CT, ultrasound, repeat CXR)."
            if case_data.get("imaging_provided") else
            "NOTE: No imaging has been provided yet. Chest X-ray may be recommended if appropriate."
        )

        user_prompt = f"""
## Patient Context
- Age: {ctx['age']}y, Sex: {ctx['sex']}
- Chief Complaint: {ctx['chief_complaint']}
- HPI: {ctx.get('hpi','')}
- Vitals: BP {ctx['vitals']['BP']}, HR {ctx['vitals']['HR']}, RR {ctx['vitals']['RR']}, SpO2 {ctx['vitals']['SpO2']}%
- Labs: D-dimer {ctx['labs'].get('D_dimer','?')} ng/mL, Troponin {ctx['labs'].get('troponin','?')} ng/mL, WBC {ctx['labs'].get('WBC','?')}, CRP {ctx['labs'].get('CRP','?')}
- Meds: {', '.join(ctx.get('meds', []))}
- PMH: {ctx.get('pmh','')}

## Imaging Status
{clinical_context_note}

## Imaging Findings (Vision Agent)
{json.dumps(imaging_findings, indent=2)}

## Radiology Report Summary
{vision_report}

## Evidence Snippets
{snippets_text if snippets_text.strip() else "No snippets retrieved. Base reasoning on clinical context only."}

Produce the DDS report JSON now.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIAGNOSIS_SYSTEM},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.2,
            timeout=45
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw.strip())

        try:
            json_match = re.search(r"\{.*\}", raw, re.S)
            report = json.loads(json_match.group()) if json_match else {}
        except (json.JSONDecodeError, AttributeError):
            report = {
                "differentials": [],
                "red_flags":     ["Unable to parse structured report — review raw output"],
                "next_steps":    ["Manual clinical review required"],
                "citations":     [],
                "_raw":          raw
            }

        # Merge snippet metadata into citations
        snippet_map = {s["id"]: s for s in snippets}
        enriched_cits = []
        for c in report.get("citations", []):
            sid = c.get("id","")
            if sid in snippet_map:
                base = snippet_map[sid]
                enriched_cits.append({
                    "id":         sid,
                    "pmid":       base.get("pmid", c.get("pmid","")),
                    "doi":        base.get("doi",  c.get("doi","")),
                    "title":      base.get("title", c.get("title","")),
                    "year":       base.get("year",  c.get("year","")),
                    "study_type": base.get("study_type", c.get("study_type","")),
                    "quote":      c.get("quote", base.get("quote",""))
                })
            else:
                enriched_cits.append(c)
        report["citations"] = enriched_cits
        return report