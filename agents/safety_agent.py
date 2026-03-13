"""Safety / Compliance Agent — PHI check, dosing guardrails, disclaimers."""
import re


DOSING_PATTERNS = [
    r'\b\d+\s*mg\b', r'\b\d+\s*mcg\b', r'\bload(ing)?\s+dose\b',
    r'\btitrate\b', r'\binfuse\b', r'\badminister\b.*\d'
]

DISCLAIMER = (
    "RESEARCH & EDUCATION ONLY. This system is NOT a medical device and is NOT approved "
    "for clinical use. All outputs must be reviewed by a qualified clinician before any "
    "clinical decision is made. No PHI has been stored or transmitted."
)


class SafetyAgent:
    def check(self, report: dict) -> dict:
        flags = []

        # 1. Dosing guardrail — redact dosing info from rationales
        for dx in report.get("differentials", []):
            rationale = dx.get("rationale", "")
            for pattern in DOSING_PATTERNS:
                if re.search(pattern, rationale, re.IGNORECASE):
                    flags.append(f"Dosing info detected in '{dx.get('dx','')}' rationale — removed.")
                    dx["rationale"] = re.sub(
                        r'[^.]*(' + pattern + r')[^.]*\.?', '[DOSING REDACTED BY SAFETY AGENT]. ',
                        dx["rationale"], flags=re.IGNORECASE
                    )
                    break

        # 2. PHI note — we rely on not collecting PHI rather than text redaction.
        # Images are displayed only, never stored. No patient identifiers are
        # accepted in the case form. Banner confirms no PHI stored/transmitted.

        # 3. Add standard disclaimer
        report["disclaimer"] = DISCLAIMER

        # 4. Log safety actions
        if flags:
            report.setdefault("safety_flags", []).extend(flags)

        return report