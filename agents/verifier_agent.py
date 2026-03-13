"""Citation Verifier Agent"""
import os
import re
from openai import OpenAI


class CitationVerifierAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def verify(self, report: dict, snippets: list[dict]) -> tuple[dict, str]:
        differentials = report.get("differentials", [])
        citations     = report.get("citations", [])
        citation_ids  = {c["id"] for c in citations}
        snippet_ids   = {s["id"] for s in snippets}

        issues = []

        # Check every differential has at least one citation
        for dx in differentials:
            support_ids = [s.get("snippet_id","") for s in dx.get("support",[])]
            valid_support = [sid for sid in support_ids if sid in citation_ids]
            if not valid_support:
                issues.append(f"'{dx.get('dx','')}' has no verified citation support.")

        # Check all citation IDs reference real snippets
        orphaned = [c["id"] for c in citations if c["id"] not in snippet_ids and snippet_ids]
        if orphaned:
            issues.append(f"Citations {orphaned} not found in retrieved snippets.")

        # Check quotes are non-empty
        empty_quotes = [c["id"] for c in citations if not c.get("quote","").strip()]
        if empty_quotes:
            issues.append(f"Citations {empty_quotes} have empty quotes.")

        if issues:
            note = "Groundedness gaps: " + "; ".join(issues) + " — Manual review recommended."
        elif not citations:
            note = "No citations retrieved. Report based on model knowledge only — verify independently."
        else:
            note = f"All {len(citations)} citations verified against {len(snippets)} retrieved snippets. ✓"

        report["groundedness_note"] = note
        return report, note
