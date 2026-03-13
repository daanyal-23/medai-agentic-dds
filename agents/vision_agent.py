"""
Vision Agent — uses GPT-4o Vision to analyze chest X-rays.
Returns structured findings JSON + bounding box overlays.
"""
import base64
import json
import io
import os
import re
from PIL import Image
from openai import OpenAI


VISION_PROMPT = """You are an expert radiologist AI assistant analyzing a chest X-ray for a decision support system (research/education only).

Analyze this chest X-ray carefully and return a JSON object. You MUST always return findings — even for normal or near-normal films, report what you observe (e.g. normal lung fields, heart size, costophrenic angles).

Return this exact structure:
{
  "findings": {
    "<finding_name>": {
      "prob": <0.0-1.0>,
      "laterality": "<left|right|bilateral|central|null>",
      "size": "<small|moderate|large|null>",
      "description": "<brief radiological description of what you see>"
    }
  },
  "overlays": [
    {
      "overlay_id": "ovl_001",
      "finding": "<finding_name>",
      "type": "bbox",
      "coords": [x_frac, y_frac, w_frac, h_frac],
      "label": "<display label>"
    }
  ],
  "report": "<2-3 sentence structured radiology report: describe lung fields, cardiac silhouette, costophrenic angles, any opacities or abnormalities>",
  "image_quality": "<adequate|limited|poor>"
}

Rules:
- ALWAYS include at least one finding — if normal, include "normal_lung_fields" with prob 0.95
- Coordinates in overlays must be fractions of image dimensions (0.0–1.0)
- Only add overlays for abnormal findings with prob > 0.4
- Consider: pneumothorax, pleural_effusion, consolidation, cardiomegaly, pulmonary_edema, pneumonia, atelectasis, nodule, opacity, normal_lung_fields
- Return ONLY valid JSON, no markdown fences, no commentary before or after
"""


class VisionAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model  = "gpt-4o"

    def _encode_image(self, image_pil: Image.Image) -> str:
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze(self, image_pil: Image.Image, case_data: dict) -> tuple[dict, list, str]:
        ctx = case_data.get("patient_context", {})
        clinical_context = (
            f"Patient: {ctx.get('age','?')}y {ctx.get('sex','?')} | "
            f"CC: {ctx.get('chief_complaint','?')} | "
            f"SpO2: {ctx.get('vitals',{}).get('SpO2','?')}% | "
            f"HR: {ctx.get('vitals',{}).get('HR','?')} bpm"
        )

        b64 = self._encode_image(image_pil)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",    "text": f"{VISION_PROMPT}\n\nClinical context: {clinical_context}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
                ]
            }],
            max_tokens=1200,
            temperature=0.1,
            timeout=30
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw.strip())

        try:
            json_match = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(json_match.group()) if json_match else {}
        except (json.JSONDecodeError, AttributeError):
            parsed = {"findings": {}, "overlays": [], "report": raw, "image_quality": "unknown"}

        findings = parsed.get("findings", {})
        overlays = parsed.get("overlays", [])
        report   = parsed.get("report", "No report generated.")

        return findings, overlays, report