#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_preprint.py
Evaluates a single preprint PDF against the GISAID rubric and produces a
structured JSON result with per-criterion scores and a final recommendation.

Usage:
    python evaluate_preprint.py --pdf papers/10.21203_rs.3.rs-8413859_v1.pdf \
        --criteria criteria.yaml --out results/10.21203_rs.3.rs-8413859_v1.json

Environment:
    OPENAI_API_KEY   required
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
import yaml
from openai import OpenAI
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# ─── constants ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
ROR_API = "https://api.ror.org/organizations"
S2_API = "https://api.semanticscholar.org/graph/v1/author/search"
S2_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
S2_AUTHOR_API = "https://api.semanticscholar.org/graph/v1/author"

# Papers over this length are trimmed to front+tail, keeping the last chunk large
# enough that results/discussion/conclusion/references (near the end) are never cut.
_FULL_THRESHOLD = 100_000
_FRONT_CAP = 60_000
_TAIL_CAP = 40_000


# ─── PDF extraction ───────────────────────────────────────────────────────────

def _extract_raw_pages(pdf_path: Path) -> list[dict]:
    pages = []
    for i, layout in enumerate(extract_pages(str(pdf_path)), start=1):
        texts = []
        for el in layout:
            if isinstance(el, LTTextContainer):
                texts.append(el.get_text())
        text = "\n".join(texts)
        text = re.sub(r"\x00|﻿| ", " ", text)
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def _select_evaluation_text(full_text: str) -> str:
    """
    Return the most evaluation-relevant portion of a paper's text.
    Papers under _FULL_THRESHOLD chars are returned whole. Larger papers keep
    the front (objective/methods) and tail (results/discussion/conclusion/
    references) and drop only the middle, so the tail is never cut off
    regardless of how long the paper is.
    """
    if len(full_text) <= _FULL_THRESHOLD:
        return full_text

    front = full_text[:_FRONT_CAP]
    tail = full_text[-_TAIL_CAP:]

    return (
        "[START OF PAPER]\n" + front
        + "\n\n[...]\n\n[END OF PAPER]\n" + tail
    )


def extract_pdf(pdf_path: Path) -> dict:
    """Return full text and metadata from a PDF."""
    pages = _extract_raw_pages(pdf_path)
    full_text = "\n\n".join(p["text"] for p in pages)
    return {
        "full_text": full_text,
        "eval_text": _select_evaluation_text(full_text),
        "header_text": full_text[:3000],  # title + authors + affiliations
        "n_pages": len(pages),
    }


# ─── external API enrichment ─────────────────────────────────────────────────

def _ror_org_name(org: dict) -> str:
    """Pick a display name from ROR v2's 'names' array (ror_display preferred)."""
    names = org.get("names", [])
    for n in names:
        if "ror_display" in n.get("types", []):
            return n.get("value", "")
    return names[0].get("value", "") if names else ""


def _ror_lookup(affiliation: str, session: requests.Session) -> dict:
    """Query ROR for a single affiliation string. Returns best match or empty dict."""
    try:
        r = session.get(ROR_API, params={"affiliation": affiliation}, timeout=10)
        if r.status_code != 200:
            return {}
        items = r.json().get("items", [])
        if not items:
            return {}
        best = items[0]
        org = best.get("organization", {})
        locations = org.get("locations", [])
        country = locations[0].get("geonames_details", {}).get("country_name", "") if locations else ""
        return {
            "name": _ror_org_name(org),
            "type": org.get("types", []),
            "country": country,
            "score": best.get("score", 0),
            "chosen": best.get("chosen", False),
        }
    except Exception:
        return {}


def _doi_from_pdf_path(pdf_path: Path) -> str:
    """Recover the DOI from a PDF filename. Same convention as
    Descargar_papers/MEJORADO/bin/descargar_pdf.py:doi_to_filename, inverted."""
    return pdf_path.stem.replace("_", "/")


def _surname(name: str) -> str:
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def _semantic_scholar_author_stats(author_id: str, session: requests.Session) -> dict:
    try:
        r = session.get(
            f"{S2_AUTHOR_API}/{author_id}",
            params={"fields": "name,paperCount,citationCount,hIndex"},
            timeout=10,
        )
        if r.status_code != 200:
            return {}
        a = r.json()
        return {
            "name": a.get("name", ""),
            "paper_count": a.get("paperCount", 0),
            "citation_count": a.get("citationCount", 0),
            "h_index": a.get("hIndex", 0),
        }
    except Exception:
        return {}


def _semantic_scholar_lookup_by_doi(doi: str, corresponding_author: str, session: requests.Session) -> dict:
    """Find the corresponding author's real Semantic Scholar authorId via the
    paper's own author list (matched by surname), then fetch their stats.
    Unambiguous when it works, but only covers papers Semantic Scholar has indexed."""
    if not doi or not corresponding_author:
        return {}
    try:
        r = session.get(f"{S2_PAPER_API}/DOI:{doi}", params={"fields": "authors"}, timeout=10)
        if r.status_code != 200:
            return {}
        authors = r.json().get("authors", [])
        target = _surname(corresponding_author)
        match = next((a for a in authors if _surname(a.get("name", "")) == target), None)
        if not match:
            return {}
        stats = _semantic_scholar_author_stats(match["authorId"], session)
        return {**stats, "match_method": "doi"} if stats else {}
    except Exception:
        return {}


def _semantic_scholar_lookup_by_name(author_name: str, session: requests.Session) -> dict:
    """Fallback when DOI lookup fails (e.g. paper not yet indexed). Searches by
    name alone, so a common name can match the wrong person — flagged as unverified."""
    try:
        r = session.get(
            S2_API,
            params={"query": author_name, "fields": "name,paperCount,citationCount,hIndex", "limit": 1},
            timeout=10,
        )
        if r.status_code != 200:
            return {}
        data = r.json().get("data", [])
        if not data:
            return {}
        a = data[0]
        return {
            "name": a.get("name", ""),
            "paper_count": a.get("paperCount", 0),
            "citation_count": a.get("citationCount", 0),
            "h_index": a.get("hIndex", 0),
            "match_method": "name_search_unverified",
        }
    except Exception:
        return {}


def _semantic_scholar_lookup(doi: str, corresponding_author: str, session: requests.Session) -> dict:
    """Look up the corresponding author's publication stats. Prefers the DOI-based
    match (unambiguous); falls back to a name-only search when the paper isn't
    indexed under that DOI yet."""
    result = _semantic_scholar_lookup_by_doi(doi, corresponding_author, session)
    if result:
        return result
    return _semantic_scholar_lookup_by_name(corresponding_author, session)


def _llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Single helper for all LLM calls."""
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def _extract_affiliations_with_llm(header_text: str, client: OpenAI, model: str) -> dict:
    """Ask Claude to extract structured author and affiliation data from the paper header."""
    prompt = f"""Extract author and affiliation information from this text (typically the first page of a scientific paper).

Return JSON with this structure:
{{
  "corresponding_author": "Name of corresponding/first author",
  "all_authors": ["Author 1", "Author 2", ...],
  "affiliations": ["Full affiliation string 1", "Full affiliation string 2", ...],
  "n_authors": 3,
  "n_institutes": 2
}}

Rules:
- List each distinct institution once in "affiliations"
- If you cannot find authors or affiliations, return empty lists
- Do not invent information

TEXT:
{header_text}"""

    raw = _llm(client, model, prompt)
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "corresponding_author": "",
            "all_authors": [],
            "affiliations": [],
            "n_authors": 0,
            "n_institutes": 0,
        }


def enrich_author_data(header_text: str, client: OpenAI, model: str, pdf_path: Path) -> dict:
    """Extract authors/affiliations and enrich with ROR and Semantic Scholar."""
    extracted = _extract_affiliations_with_llm(header_text, client, model)

    session = requests.Session()
    session.headers["User-Agent"] = "GISAID-preprint-evaluator/1.0 (mailto:juanfinello@gmail.com)"

    ror_results = []
    for aff in extracted.get("affiliations", [])[:6]:  # cap at 6 to avoid rate limits
        result = _ror_lookup(aff, session)
        if result:
            ror_results.append(result)
        time.sleep(0.3)

    s2_result = {}
    corr = extracted.get("corresponding_author", "")
    if corr:
        doi = _doi_from_pdf_path(pdf_path)
        s2_result = _semantic_scholar_lookup(doi, corr, session)
        time.sleep(0.3)

    return {
        "extracted": extracted,
        "ror_results": ror_results,
        "semantic_scholar": s2_result,
    }


# ─── LLM evaluation ──────────────────────────────────────────────────────────

def _build_content_prompt(full_text: str, criteria: dict) -> str:
    rqm = criteria["research_question_and_methods"]
    rac = criteria["results_and_conclusion"]
    ref = criteria["references"]

    return f"""You are evaluating a scientific preprint for GISAID eligibility.
Read the full paper text below and score THREE criteria using ONLY the evidence present.

=== CRITERION 1: {rqm["label"]} ===
Score 1: {rqm["score_1"]["description"]}
Score 2: {rqm["score_2"]["description"]}
Score 3: {rqm["score_3"]["description"]}

=== CRITERION 2: {rac["label"]} ===
Score 1: {rac["score_1"]["description"]}
Score 2: {rac["score_2"]["description"]}
Score 3: {rac["score_3"]["description"]}

=== CRITERION 3: {ref["label"]} ===
Score 1: {ref["score_1"]["description"]}
Score 2: {ref["score_2"]["description"]}
Score 3: {ref["score_3"]["description"]}

=== HOW TO SCORE EACH CRITERION ===
For each of the three criteria, follow this procedure before committing to a score:
1. Check the score_3 description first. Does the paper clearly meet it? If yes, score 3 — stop.
2. If not, check the score_1 description. Does the paper clearly show those weaknesses? If yes, score 1 — stop.
3. Only score 2 if the paper genuinely sits between the two — some strong elements
   present, but not enough for a clear 3, with no red flags severe enough for a clear 1.

Score 2 is not a safe default. It should be your conclusion in a minority of cases,
not most of them. A real, diverse batch of preprints spans weak, moderate, and strong
work — if you are about to assign 2 to all three criteria for this paper, re-read the
score_1 and score_3 descriptions again before finalizing; you are likely
under-differentiating.

- Quote must be a verbatim phrase copied from the paper (max 220 characters).
- Justify in 1–2 sentences, naming which score_1/score_2/score_3 description the
  evidence matches.
- For references: carefully count the total number of entries in the reference list
  and report it as "reference_count" (an integer). The score for this criterion is
  derived automatically from that count, not chosen by you — focus entirely on
  counting accurately; do not try to reason about what score the count "should" map to.

=== FULL PAPER TEXT ===
{full_text}

Return ONLY this JSON (no markdown, no extra text):
{{
  "research_question_and_methods": {{
    "score": 3,
    "justification": "brief explanation grounded in the text, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "results_and_conclusion": {{
    "score": 1,
    "justification": "brief explanation grounded in the text, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "references": {{
    "reference_count": 24,
    "justification": "brief explanation of reference quality and coverage",
    "quote": "sample reference from the paper"
  }}
}}"""


def _build_author_prompt(author_data: dict, criteria: dict) -> str:
    extracted = author_data["extracted"]
    ror = author_data["ror_results"]
    s2 = author_data["semantic_scholar"]

    crit = criteria["author_credibility"]

    ror_summary = json.dumps(ror, indent=2) if ror else "No ROR data found"
    s2_summary = json.dumps(s2, indent=2) if s2 else "No Semantic Scholar data found"

    return f"""You are evaluating the authors and institutions of a scientific preprint for GISAID eligibility.

=== CRITERION: {crit["label"]} ===
Score 1: {crit["score_1"]["description"]}
Score 2: {crit["score_2"]["description"]}
Score 3: {crit["score_3"]["description"]}

=== EXTRACTED AUTHOR DATA ===
Corresponding/first author: {extracted.get("corresponding_author", "unknown")}
All authors ({extracted.get("n_authors", "?")}): {", ".join(extracted.get("all_authors", [])[:8])}
Distinct affiliations ({extracted.get("n_institutes", "?")}):
{chr(10).join("- " + a for a in extracted.get("affiliations", []))}

=== INSTITUTION DATA (ROR API) ===
{ror_summary}

=== CORRESPONDING AUTHOR (Semantic Scholar) ===
{s2_summary}

=== INSTRUCTIONS ===
- Base the score on institution type (prefer universities/research institutes over commercial entities)
- Use ROR "type" field: "Education" or "Research" institutions are credible; "Company" or missing data is a flag
- Use Semantic Scholar to judge author expertise (paper_count > 10 and h_index > 3 suggests established researcher)
- Check "match_method": "doi" means the Semantic Scholar record was matched via the paper's own author list (reliable);
  "name_search_unverified" means it was matched by name alone and could belong to a different person with the same name —
  treat it as a weak signal, not a confirmed match
- If ROR or Semantic Scholar returned no data, note the uncertainty but do not automatically penalize

Return ONLY this JSON (no markdown, no extra text):
{{
  "author_credibility": {{
    "score": 2,
    "justification": "brief explanation",
    "institutions_assessed": ["list of institution names"],
    "n_institutes": 2,
    "corresponding_author_papers": 0,
    "flags": ["any concerns found, e.g. commercial affiliation"]
  }}
}}"""


def _score_from_reference_count(n: int) -> int:
    """Deterministic score from a reference count — the LLM only has to count,
    not compare against the threshold (it was getting the comparison wrong,
    e.g. calling 58 'within 10-20')."""
    if n > 20:
        return 3
    if n < 10:
        return 1
    return 2


def evaluate_content(full_text: str, criteria: dict, client: OpenAI, model: str) -> dict:
    """Evaluate criteria 2, 3, 4 (research question, results, references) from full paper text."""
    prompt = _build_content_prompt(full_text, criteria)
    raw = _llm(client, model, prompt)
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        result = json.loads(raw)
    except Exception:
        return {
            "research_question_and_methods": {"score": 1, "justification": "Parse error", "quote": ""},
            "results_and_conclusion": {"score": 1, "justification": "Parse error", "quote": ""},
            "references": {"score": 1, "justification": "Parse error", "quote": ""},
        }

    ref = result.get("references", {})
    ref_count = ref.get("reference_count")
    if isinstance(ref_count, int):
        ref["score"] = _score_from_reference_count(ref_count)
    else:
        ref["score"] = 1
        ref["justification"] = (ref.get("justification", "") + " [no reference_count reported]").strip()
    return result


def evaluate_author_credibility(author_data: dict, criteria: dict, client: OpenAI, model: str) -> dict:
    """Evaluate criterion 1 (author credibility) using enriched author data."""
    prompt = _build_author_prompt(author_data, criteria)
    raw = _llm(client, model, prompt, max_tokens=512)
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "author_credibility": {"score": 1, "justification": "Parse error", "flags": []}
        }


# ─── quote validation ────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    # collapse typographic variants so curly/straight quotes and dashes compare equal
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[''`]", "'", text)          # curly single quotes → '
    text = re.sub(r'[""„«»]', '"', text)        # curly double quotes → "
    text = re.sub(r"[–—−]", "-", text)          # dashes → hyphen
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _quote_found(quote: str, full_text: str) -> bool:
    """Return True if quote (or its first 12 words) appears in full_text."""
    q = _norm(quote)
    if not q:
        return False
    if q in _norm(full_text):
        return True
    # partial match: first 12 words (handles truncation artifacts)
    words = q.split()
    if len(words) >= 8:
        partial = " ".join(words[:12])
        return partial in _norm(full_text)
    return False


def validate_quotes(criterion_results: dict, full_text: str) -> dict:
    """
    For each criterion with a 'quote' field, verify the quote exists verbatim
    in full_text. Adds 'quote_valid' bool to each criterion. Downgrades score
    from 3 → 2 when the supporting quote cannot be verified.
    """
    CRITERIA_WITH_QUOTES = [
        "research_question_and_methods",
        "results_and_conclusion",
        "references",
    ]
    for key in CRITERIA_WITH_QUOTES:
        if key not in criterion_results:
            continue
        c = criterion_results[key]
        quote = c.get("quote", "")
        valid = _quote_found(quote, full_text) if quote else False
        c["quote_valid"] = valid
        if not valid and c.get("score") == 3:
            c["score"] = 2
            c["score_note"] = "downgraded 3→2: supporting quote not found verbatim in text"
    return criterion_results


# ─── scoring ─────────────────────────────────────────────────────────────────

def compute_scores(criterion_results: dict) -> dict:
    """
    Aggregate per-criterion scores.
    feedback is excluded from automatic scoring (requires manual check).
    Recommendation thresholds are not applied here — calibrate with real cases.
    """
    scored_criteria = ["author_credibility", "research_question_and_methods",
                       "results_and_conclusion", "references"]

    scores = {}
    for k in scored_criteria:
        if k in criterion_results:
            scores[k] = criterion_results[k].get("score", 0)

    valid_scores = [s for s in scores.values() if isinstance(s, int) and s > 0]
    total = sum(valid_scores)
    max_possible = len(scored_criteria) * 3
    avg = round(total / len(valid_scores), 2) if valid_scores else 0

    return {
        "per_criterion": scores,
        "total": total,
        "max": max_possible,
        "average": avg,
        "n_criteria_evaluated": len(valid_scores),
        "recommendation": _recommendation(avg, scores),
    }


def _recommendation(avg: float, scores: dict) -> str:
    """
    Provisional recommendation based on average score and any score=1 flags.
    Thresholds are approximate — calibrate with real labeled cases.
    """
    n_ones = sum(1 for s in scores.values() if s == 1)

    if avg >= 2.4 and n_ones == 0:
        return "accept"
    elif avg >= 1.7 and n_ones <= 1:
        return "accept_with_reservations"
    else:
        return "reject"


# ─── main pipeline ────────────────────────────────────────────────────────────

def evaluate_preprint(pdf_path: Path, criteria_path: Path, model: str, client: OpenAI) -> dict:
    """Full evaluation pipeline for a single PDF."""
    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))["criteria"]

    # Step 1: extract PDF
    pdf_data = extract_pdf(pdf_path)

    # Step 2: enrich authors (PDF + ROR + Semantic Scholar)
    author_data = enrich_author_data(pdf_data["header_text"], client, model, pdf_path)

    # Step 3: evaluate content criteria (2, 3, 4) in one LLM call
    content_results = evaluate_content(pdf_data["eval_text"], criteria, client, model)

    # Step 4: evaluate author credibility (1) in one LLM call
    author_results = evaluate_author_credibility(author_data, criteria, client, model)

    # Merge all criterion results
    criterion_results = {**author_results, **content_results}
    criterion_results["feedback"] = {
        "score": None,
        "justification": "Not evaluated automatically. Requires manual check of preprint server.",
    }

    # Step 5: validate quotes against full text (catches hallucinated evidence)
    criterion_results = validate_quotes(criterion_results, pdf_data["full_text"])

    # Step 6: aggregate scores
    scoring = compute_scores(criterion_results)

    # Add quote health summary to scoring
    checked = ["research_question_and_methods", "results_and_conclusion", "references"]
    invalid = [k for k in checked if not criterion_results.get(k, {}).get("quote_valid", True)]
    scoring["quotes_invalid"] = invalid

    return {
        "source_pdf": pdf_path.name,
        "model": model,
        "n_pages": pdf_data["n_pages"],
        "author_enrichment": author_data,
        "criteria": criterion_results,
        "scoring": scoring,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate a preprint PDF for GISAID eligibility.")
    ap.add_argument("--pdf", required=True, help="Path to the preprint PDF")
    ap.add_argument("--criteria", default="criteria.yaml", help="Path to criteria.yaml")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to use")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    criteria_path = Path(args.criteria)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not criteria_path.exists():
        raise FileNotFoundError(f"criteria.yaml not found: {criteria_path}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    result = evaluate_preprint(pdf_path, criteria_path, args.model, client)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    scoring = result["scoring"]
    print(f"✓ {pdf_path.name}")
    print(f"  Scores: {scoring['per_criterion']}")
    print(f"  Average: {scoring['average']} / 3.0  |  Recommendation: {scoring['recommendation']}")
    invalid = scoring.get("quotes_invalid", [])
    if invalid:
        print(f"  ⚠ Quotes not verified: {', '.join(invalid)}")
    print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
