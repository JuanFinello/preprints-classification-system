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
import base64
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

# gpt-4o-mini retired from ChatGPT Feb 2026 and dropped off the current pricing
# table; gpt-5.6-terra is the same price tier as its old gpt-5.4 sibling but the
# newer generation, and (like gpt-4o+) parses PDFs as text+page-images natively.
DEFAULT_MODEL = "gpt-5.6-terra"
ROR_API = "https://api.ror.org/organizations"
ROR_API_V2_ORG = "https://api.ror.org/v2/organizations"
OPENALEX_API = "https://api.openalex.org/works/doi:"
S2_API = "https://api.semanticscholar.org/graph/v1/author/search"
S2_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
S2_AUTHOR_API = "https://api.semanticscholar.org/graph/v1/author"
CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_MAILTO = "juanfinello@gmail.com"  # joins Crossref's "polite pool": 3 req/s instead of 1 req/s anonymous, per their own etiquette guidance
PREREVIEW_BASE = "https://prereview.org"

# Papers over this length are trimmed to front+tail, keeping the last chunk large
# enough that results/discussion/conclusion/references (near the end) are never cut.
# The content call carries a whole paper and reasons at effort "high"; reasoning
# tokens are invisible but come out of this same budget, so a cap that only fits
# the JSON truncates the answer mid-object and the parse fails. 4500 did exactly
# that on 5 of the first 18 runs.
_CONTENT_MAX_TOKENS = 16000
_AUTHOR_MAX_TOKENS = 4000

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


def _ror_aliases(ror_id: str, session: requests.Session) -> list[str]:
    """Fetch every alternate full name ROR has on file for this institution ID
    (aliases, labels in other languages, its ror_display name) — deliberately
    excludes the "acronym" type (e.g. "MRD-G"), since a short acronym is prone
    to matching unrelated text by coincidence even with word-boundary checks.

    Used as a second chance when OpenAlex's own chosen display_name doesn't
    literally appear in the paper's own affiliation text for an author (real
    case: OpenAlex calls it "Fred Hutch Cancer Center", the paper itself says
    "Fred Hutchinson Cancer Center" — same ROR ID, and ROR has the full name
    on file as a registered alias). This only ever validates text the paper
    itself already contains against another known real name of the SAME
    registered entity — it never pulls in an institution the paper didn't
    mention (that would just fail every alias too, same as it fails the
    primary name)."""
    if not ror_id:
        return []
    try:
        r = session.get(f"{ROR_API_V2_ORG}/{ror_id}", timeout=10)
        if r.status_code != 200:
            return []
        names = r.json().get("names", [])
        return [n["value"] for n in names if "acronym" not in n.get("types", [])]
    except Exception:
        return []


def _split_affiliations(affiliations: list[str]) -> list[str]:
    """Split affiliation strings that bundle multiple institutions with ';'
    (common in headers listing an author's institute + parent academy together)
    before they get queried against ROR one at a time — otherwise ROR's fuzzy
    matcher tends to return only the more generic/larger institution and silently
    drop the more specific one."""
    result = []
    for aff in affiliations:
        parts = [p.strip() for p in aff.split(";") if p.strip()]
        result.extend(parts if parts else [aff])
    return result


def _openalex_institutions_by_doi(doi: str, session: requests.Session) -> dict:
    """Fetch author->institution links from OpenAlex's own record for this DOI,
    verified against the paper's own raw affiliation text.

    OpenAlex resolves each author's raw affiliation string against its own
    institution index, and that resolution step can itself mis-split or
    mismatch (observed on a real paper: a garbled multi-institution Chinese
    affiliation string got matched to an unrelated "King Center" entity that
    never appears anywhere in the actual text). So a matched institution is
    only kept if its name is found, normalized, inside that same author's
    raw_affiliation_strings — otherwise it's dropped and flagged instead of
    trusted blindly just because it came from a DOI-anchored API. Before
    dropping, it gets one more chance against that institution's own ROR
    aliases (see _ror_aliases) — OpenAlex's chosen display_name is sometimes
    a nickname the paper itself doesn't use (real case: OpenAlex says "Fred
    Hutch Cancer Center", the paper says "Fred Hutchinson Cancer Center",
    same ROR ID, and ROR has the full name on file). This only re-checks the
    paper's own already-declared text against another known real name of the
    same registered entity — it can't resurrect an institution the paper
    never mentioned at all (that still fails every alias too).

    Also extracts the paper's own top research fields (for author_expertise
    field-matching, see _openalex_author_field_match) and, per author, whether
    OpenAlex's resolved identity actually matches the name printed in the
    paper's own byline (raw_author_name) — OpenAlex's author disambiguation
    can attach the wrong person entirely (observed: a paper's real "Miao Mei"
    got resolved to an unrelated "May Lei Mei" profile with 141 papers, all
    in dentistry). identity_verified=False on an author means none of their
    OpenAlex-derived data (institutions here, topics elsewhere) should be
    trusted as theirs.

    Returns {"institutions": [...], "warnings": [...], "paper_fields": [...],
    "authors": [{"name", "id", "identity_verified"}, ...]}.
    """
    empty = {"institutions": [], "warnings": [], "paper_fields": [], "authors": []}
    if not doi:
        return empty
    try:
        r = session.get(f"{OPENALEX_API}{requests.utils.quote(doi, safe='')}", timeout=10)
        if r.status_code != 200:
            return empty
        data = r.json()
    except Exception:
        return empty

    paper_fields = []
    for t in data.get("topics", [])[:5]:
        field_name = (t.get("field") or {}).get("display_name", "")
        if field_name and field_name not in paper_fields:
            paper_fields.append(field_name)

    results = []
    warnings = []
    authors_info = []
    ror_alias_cache: dict[str, list[str]] = {}
    for a in data.get("authorships", []):
        author = a.get("author") or {}
        author_name = author.get("display_name", "")
        author_id = author.get("id", "")
        raw_name = a.get("raw_author_name", "")

        identity_verified = None
        if raw_name and author_name:
            # subset (not equality) so a resolved middle name/initial ("Gerald
            # Kellar" -> "Gerald G. Kellar") doesn't get flagged as a mismatch —
            # only a name with no overlap at all (e.g. "Miao Mei" -> "May Lei
            # Mei") counts as one.
            raw_words = set(_norm(raw_name).split())
            auth_words = set(_norm(author_name).split())
            identity_verified = raw_words <= auth_words or auth_words <= raw_words
            if not identity_verified:
                warnings.append(
                    f"OpenAlex matched the paper's byline name '{raw_name}' to a "
                    f"different author identity '{author_name}' — that author's "
                    f"OpenAlex-derived data may belong to the wrong person."
                )
        authors_info.append({
            "name": author_name,
            "id": author_id,
            "identity_verified": identity_verified,
            "position": a.get("author_position", ""),  # "first" | "middle" | "last"
        })

        raw_text = _norm(" ".join(a.get("raw_affiliation_strings", [])))
        for inst in a.get("institutions", []):
            name = inst.get("display_name", "")
            if not name:
                continue
            entry = {
                "author": author_name,
                "position": a.get("author_position", ""),  # "first" | "middle" | "last"
                "name": name,
                "type": [inst.get("type", "")] if inst.get("type") else [],
                "country": inst.get("country_code", ""),
                "ror": inst.get("ror", ""),
            }
            if not raw_text:
                warnings.append(
                    f"OpenAlex gave no raw affiliation text for '{author_name}' "
                    f"to verify '{name}' against — kept, but unverified."
                )
                results.append(entry)
            elif re.search(r"\b" + re.escape(_norm(name)) + r"\b", raw_text):
                results.append(entry)
            else:
                ror_id = inst.get("ror", "")
                if ror_id and ror_id not in ror_alias_cache:
                    ror_alias_cache[ror_id] = _ror_aliases(ror_id, session)
                matched_alias = next(
                    (alias for alias in ror_alias_cache.get(ror_id, [])
                     if re.search(r"\b" + re.escape(_norm(alias)) + r"\b", raw_text)),
                    None,
                )
                if matched_alias:
                    warnings.append(
                        f"'{name}' matched '{author_name}' via a ROR alias "
                        f"('{matched_alias}') rather than OpenAlex's own display "
                        f"name — kept."
                    )
                    results.append(entry)
                else:
                    warnings.append(
                        f"OpenAlex matched '{name}' to '{author_name}' but that name "
                        f"(nor any of its known ROR aliases) doesn't appear anywhere "
                        f"in the paper's own affiliation text — dropped as a likely "
                        f"mismatch."
                    )
    return {"institutions": results, "warnings": warnings, "paper_fields": paper_fields, "authors": authors_info}


def _match_author_name(name_a: str, name_b: str) -> bool:
    """Cross-API name match (Semantic Scholar vs OpenAlex spell/order names
    differently, e.g. 'W. Tan' vs 'Wenjie Tan'). Requires the surname to match
    AND the first name to be compatible (equal, or one is an initial of the
    other) — matching on surname alone produces false collisions when a paper
    has multiple authors sharing it (real case: this function used to match
    both 'W. Tan' and 'Xu Tan' from Semantic Scholar to the SAME 'Jiali Tan'
    from OpenAlex, because all three share the surname 'Tan' and surname-only
    matching just grabs whichever one comes first). Also tries the second
    name's tokens reversed, since East Asian names get romanized as either
    family-name-first or given-name-first inconsistently across APIs."""
    def tokens(n):
        return [t.rstrip(".") for t in _norm(n).split()]

    def first_name_compatible(fa, fb):
        return fa == fb or (len(fa) == 1 and fb.startswith(fa)) or (len(fb) == 1 and fa.startswith(fb))

    a, b = tokens(name_a), tokens(name_b)
    if len(a) < 2 or len(b) < 2:
        return False
    if a[-1] == b[-1] and first_name_compatible(a[0], b[0]):
        return True
    b_rev = b[::-1]
    return a[-1] == b_rev[-1] and first_name_compatible(a[0], b_rev[0])


def _openalex_author_field_match(authors: list[dict], paper_fields: list[str],
                                  session: requests.Session) -> dict:
    """For each OpenAlex author id (from _openalex_institutions_by_doi), check
    whether their own top research fields overlap with this paper's fields.

    Catches authors whose overall publication record is large but not evidently
    in this paper's area — real publication volume isn't the same as expertise
    "in the field" the rubric asks for. Also catches identity mismatches: an
    author flagged identity_verified=False here always comes back field_match
    unknown rather than trusting a stranger's topics.

    Also pulls each author's works_count/h_index/cited_by_count from the same
    OpenAlex profile — free, since the batch call already returns the full
    author object. Used in enrich_author_data to backfill authors Semantic
    Scholar has no record for at all (observed: a ResearchSquare preprint not
    yet indexed by S2 still had full OpenAlex author profiles for everyone).

    Returns {"field_match": {author_name: {"field_match": bool|None,
    "top_fields": [...], "works_count": int|None, "h_index": int|None,
    "citation_count": int|None}}, "warnings": [...]}.
    """
    warnings = []
    field_match = {}
    paper_field_set = set(paper_fields)

    valid = [a for a in authors if a.get("id") and a.get("identity_verified") is not False]
    ids = [a["id"] for a in valid]

    profiles = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        try:
            r = session.get(
                "https://api.openalex.org/authors",
                params={"filter": f"openalex_id:{'|'.join(batch)}"},
                timeout=15,
            )
            if r.status_code == 200:
                for a in r.json().get("results", []):
                    profiles[a.get("id", "")] = a
        except Exception:
            pass

    for a in authors:
        name = a["name"]
        position = a.get("position", "")
        if a.get("identity_verified") is False or not a.get("id") or a["id"] not in profiles:
            field_match[name] = {
                "field_match": None, "top_fields": [], "position": position,
                "works_count": None, "h_index": None, "citation_count": None,
            }
            continue
        profile = profiles[a["id"]]
        topics = profile.get("topics", [])[:5]
        fields = [f for f in ((t.get("field") or {}).get("display_name", "") for t in topics) if f]
        overlap = bool(paper_field_set & set(fields)) if paper_field_set and fields else None
        field_match[name] = {
            "field_match": overlap,
            "top_fields": fields,
            "position": position,
            "works_count": profile.get("works_count"),
            "h_index": (profile.get("summary_stats") or {}).get("h_index"),
            "citation_count": profile.get("cited_by_count"),
        }
        if overlap is False:
            warnings.append(
                f"'{name}': top research fields ({', '.join(fields[:3])}) don't overlap "
                f"with this paper's fields ({', '.join(paper_fields)}) — publication "
                f"volume may not reflect expertise in THIS field."
            )

    return {"field_match": field_match, "warnings": warnings}


def _doi_from_pdf_path(pdf_path: Path) -> str:
    """Recover the DOI from a PDF filename. Same convention as
    Descargar_papers/MEJORADO/bin/descargar_pdf.py:doi_to_filename, inverted."""
    return pdf_path.stem.replace("_", "/")


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


def _semantic_scholar_paper_authors(doi: str, session: requests.Session) -> list[dict]:
    """Fetch the author list Semantic Scholar has on file for this exact DOI.
    Each entry already carries its own authorId, resolved by S2 against this
    specific paper — unambiguous, unlike a name search."""
    if not doi:
        return []
    try:
        r = session.get(f"{S2_PAPER_API}/DOI:{doi}", params={"fields": "authors"}, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("authors", [])
    except Exception:
        return []


def _semantic_scholar_lookup_by_name(author_name: str, session: requests.Session) -> dict:
    """Fallback when the paper isn't indexed under its DOI. Searches by name
    alone, so a common name can match the wrong person — flagged as unverified."""
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


def _semantic_scholar_lookup_all_authors(doi: str, corresponding_author: str, session: requests.Session) -> dict:
    """Fetch publication stats for every author Semantic Scholar lists on this
    paper's own DOI record — each authorId comes pre-resolved by S2 against this
    specific paper, so it's as reliable as the old single-author DOI lookup, just
    for everyone instead of one person. Falls back to a name-only search for the
    corresponding author alone when the paper isn't indexed under its DOI.

    Returns {"authors": [stats, ...], "warnings": [str, ...]}. author_credibility
    depends entirely on this data (the LLM never sees the PDF for that criterion),
    so any gap here directly limits how much of the rubric can actually be judged
    — warnings surface that instead of failing silently.
    """
    warnings = []
    paper_authors = _semantic_scholar_paper_authors(doi, session)

    if not paper_authors:
        warnings.append(
            "Semantic Scholar has no record for this DOI — falling back to a "
            "name-only search for the corresponding author only; every other "
            "author's expertise is unverified."
        )
        fallback = _semantic_scholar_lookup_by_name(corresponding_author, session) if corresponding_author else {}
        if corresponding_author and not fallback:
            warnings.append(
                f"Semantic Scholar name search also failed for corresponding "
                f"author '{corresponding_author}' — no author expertise data "
                f"available at all for this paper."
            )
        return {"authors": [fallback] if fallback else [], "warnings": warnings}

    stats_list = []
    for a in paper_authors:
        author_id = a.get("authorId")
        name = a.get("name", "")
        if not author_id:
            warnings.append(f"Semantic Scholar: no authorId for '{name}' — skipped.")
            continue
        stats = _semantic_scholar_author_stats(author_id, session)
        time.sleep(0.3)
        if not stats:
            warnings.append(f"Semantic Scholar: stats request failed for '{name}'.")
            continue
        stats["match_method"] = "doi"
        stats_list.append(stats)

    if len(stats_list) < len(paper_authors):
        warnings.append(
            f"Semantic Scholar: only {len(stats_list)}/{len(paper_authors)} "
            f"authors on record got verified stats — author_credibility for "
            f"this paper is scored on incomplete data."
        )

    return {"authors": stats_list, "warnings": warnings}


def _llm_call(client: OpenAI, model: str, messages: list, max_tokens: int,
              reasoning_effort: str | None = None) -> tuple[str, str | None, object]:
    """Every LLM call goes through here. Returns the text alongside the
    finish_reason and usage, which are what tell a truncated answer apart from
    a malformed one when the caller fails to parse the result."""
    kwargs = {}
    if reasoning_effort is not None:
        # gpt-5.x reasoning tokens are invisible but come out of the same
        # max_completion_tokens budget - see evaluate_author_credibility for
        # a case that silently truncated to 0 visible tokens without this.
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,  # gpt-5.x rejects the old max_tokens param
        messages=messages,
        **kwargs,
    )
    choice = resp.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason, resp.usage


def _llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 1024,
         reasoning_effort: str | None = None) -> str:
    """Single helper for all LLM calls that only need the text back."""
    text, _, _ = _llm_call(client, model, [{"role": "user", "content": prompt}],
                           max_tokens, reasoning_effort)
    return text


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
    """Extract authors/affiliations and enrich with institution + author data.

    Institutions: tries OpenAlex first (DOI-anchored, keeps the author link,
    real ROR ids, no cap on how many come back). Falls back to LLM-extracted
    header affiliations queried against ROR one by one when OpenAlex has no
    record for this DOI — every affiliation is tried (no cap), compound
    "X; Y" strings are split first, and any lookup that comes back empty is
    recorded as a warning instead of silently dropped.

    Authors: Semantic Scholar, see _semantic_scholar_lookup_all_authors.
    """
    extracted = _extract_affiliations_with_llm(header_text, client, model)

    session = requests.Session()
    session.headers["User-Agent"] = "GISAID-preprint-evaluator/1.0 (mailto:juanfinello@gmail.com)"

    doi = _doi_from_pdf_path(pdf_path)
    warnings = []

    openalex = _openalex_institutions_by_doi(doi, session)
    ror_results = openalex["institutions"]
    warnings.extend(openalex["warnings"])

    if ror_results:
        institution_source = "openalex"
    else:
        institution_source = "ror_fallback"
        warnings.append(
            "OpenAlex had no verified institution data for this DOI — falling "
            "back to header-text affiliation extraction + ROR lookup (the "
            "author-institution link is lost in this path; institutions are "
            "matched as a flat list)."
        )
        affiliations = _split_affiliations(extracted.get("affiliations", []))
        for aff in affiliations:
            result = _ror_lookup(aff, session)
            if result:
                ror_results.append(result)
            else:
                warnings.append(f"ROR: no match found for affiliation '{aff}'.")
            time.sleep(0.3)

    corr = extracted.get("corresponding_author", "")
    s2_data = _semantic_scholar_lookup_all_authors(doi, corr, session)
    warnings.extend(s2_data["warnings"])

    field_match = _openalex_author_field_match(
        openalex.get("authors", []), openalex.get("paper_fields", []), session
    )
    warnings.extend(field_match["warnings"])

    covered = set()
    for author_stat in s2_data["authors"]:
        match_name = next(
            (name for name in field_match["field_match"]
             if _match_author_name(author_stat.get("name", ""), name)),
            None,
        )
        fm = field_match["field_match"].get(match_name, {})
        author_stat["field_match"] = fm.get("field_match") if match_name else None
        author_stat["position"] = fm.get("position", "") if match_name else ""
        if match_name:
            covered.add(match_name)

    # Backfill authors Semantic Scholar has no record for at all, using
    # OpenAlex's own works_count/h_index — covers preprint servers (e.g.
    # ResearchSquare) that S2 often hasn't indexed yet but OpenAlex already has.
    n_backfilled = 0
    for name, fm in field_match["field_match"].items():
        if name in covered or fm.get("works_count") is None:
            continue
        s2_data["authors"].append({
            "name": name,
            "paper_count": fm["works_count"],
            "citation_count": fm.get("citation_count"),
            "h_index": fm.get("h_index"),
            "match_method": "openalex_fallback",
            "field_match": fm["field_match"],
            "position": fm.get("position", ""),
        })
        n_backfilled += 1
    if n_backfilled:
        warnings.append(
            f"Semantic Scholar had no record for {n_backfilled} author(s) — "
            f"their publication stats were backfilled from OpenAlex instead."
        )

    return {
        "extracted": extracted,
        "ror_results": ror_results,
        "institution_source": institution_source,
        "semantic_scholar": s2_data,
        "paper_fields": openalex.get("paper_fields", []),
        "data_quality_warnings": warnings,
    }


# ─── LLM evaluation ──────────────────────────────────────────────────────────

def _build_content_prompt(full_text: str, criteria: dict) -> str:
    rqm = criteria["research_question_and_methods"]
    sub = rqm["sub_criteria"]
    obj_c = sub["objective_and_hypothesis"]
    phr_c = sub["public_health_relevance"]
    dr_c = sub["study_design_rigor"]
    rac = criteria["results_and_conclusion"]
    ref = criteria["references"]

    return f"""You are evaluating a scientific preprint for GISAID eligibility.
Read the full paper text below and score the criteria below using ONLY the evidence present.

=== CRITERION 1: {rqm["label"]} — 3 INDEPENDENT SUB-CRITERIA ===
Score each sub-criterion on its own evidence. A weak result on one must NOT pull
down your score on another.

--- Sub-criterion 1a: {obj_c["label"]} ---
Score 1: {obj_c["score_1"]["description"]}
Score 2: {obj_c["score_2"]["description"]}
Score 3: {obj_c["score_3"]["description"]}

--- Sub-criterion 1b: {phr_c["label"]} ---
Score 1: {phr_c["score_1"]["description"]}
Score 2: {phr_c["score_2"]["description"]}
Score 3: {phr_c["score_3"]["description"]}

--- Sub-criterion 1c: {dr_c["label"]} ---
Score 1: {dr_c["score_1"]["description"]}
Score 2: {dr_c["score_2"]["description"]}
Score 3: {dr_c["score_3"]["description"]}

=== CRITERION 2: {rac["label"]} ===
Score 1: {rac["score_1"]["description"]}
Score 2: {rac["score_2"]["description"]}
Score 3: {rac["score_3"]["description"]}

=== CRITERION 3: {ref["label"]} ===
Score 1: {ref["score_1"]["description"]}
Score 2: {ref["score_2"]["description"]}
Score 3: {ref["score_3"]["description"]}

=== HOW TO SCORE EACH (SUB-)CRITERION ===
For each of the 5 things you are scoring (3 sub-criteria of criterion 1, plus
criteria 2 and 3), follow this procedure before committing to a score:
1. Check the score_3 description first. Does the paper clearly meet it? If yes, score 3 — stop.
2. If not, check the score_1 description. Does the paper clearly show those weaknesses? If yes, score 1 — stop.
3. Only score 2 if the paper genuinely sits between the two — some strong elements
   present, but not enough for a clear 3, with no red flags severe enough for a clear 1.

Score 2 is not a safe default. It should be your conclusion in a minority of cases,
not most of them. A real, diverse batch of preprints spans weak, moderate, and strong
work — if you are about to assign 2 to everything for this paper, re-read the
score_1 and score_3 descriptions again before finalizing; you are likely
under-differentiating.

You are working from extracted text only — some tables survive here as unstructured
runs of numbers (headers separated from their values), and figures/images are not
included at all. Do not conclude "lacks supporting figures or tables" just because
you cannot see them visually — look for whether tabular or numeric data survived in
the text and judge internal consistency from that. Only cite an actual absence of
figures/tables if there is truly no such data anywhere in the text.

- Quote must be a verbatim phrase copied from the paper (max 220 characters), one
  per sub-criterion/criterion.
- Justify in 1–2 sentences, naming which score_1/score_2/score_3 description the
  evidence matches.
- For results_and_conclusion: actively cross-check numeric values across tables,
  figures, and the surrounding text for internal consistency — do not just check
  whether tables/figures are present. Specifically: (1) flag any count of discrete
  items (reads, sequences, samples) reported with a decimal/fraction — that is
  intrinsically impossible and signals a data-integrity problem, not merely
  "unclear parameters"; (2) check whether a summary statistic stated in the text
  (a mean, a total) matches the values in the table it is summarizing; (3) if a
  specific data point directly contradicts a conclusion the paper itself states
  (e.g., the paper claims genome integrity/concordance "was well preserved" but
  its own reported identity numbers show a large drop for one sample), that is
  direct evidence for score_1's "conclusions ... not grounded in the evidence
  presented" — do not soften this to score_2's "unclear statistical parameters",
  which is for vague or incomplete reporting, not a documented contradiction
  between the paper's own data and its own conclusion.
- For references: report facts, not a score. The score is computed in code from what
  you report, so report each field carefully and do not try to reason about the final
  number.
  (a) "reference_count": the total number of entries in the reference list, as an
  integer. Count them; do not estimate.
  (b) "off_topic_references": a list of the entries that are genuinely unrelated to
  this paper's subject matter (e.g. a paper about virus X citing a genome-assembly
  study of an unrelated fungus, or a plant-virology review, with no clear
  methodological link). Identify each by number and first author, with a few words
  on why. Return an empty list if there are none — an empty list is the expected
  answer for a well-focused bibliography, so do not manufacture entries, but do not
  overlook a real one either: a single off-topic entry caps this criterion below the
  top score.
  (c) "non_peer_reviewed_count": how many entries are preprints, blog posts, press
  releases or other non-peer-reviewed sources, as an integer.
  (d) "self_citation_count": how many entries are by this paper's own authors, as an
  integer.
  Write the justification about source quality and coverage (peer-reviewed vs. not,
  foundational vs. recent balance, domain relevance, self-citation reliance), naming
  the off-topic entries you found. Do not write about whether the count itself feels
  low or high, since that judgment is not used.

=== FULL PAPER TEXT ===
{full_text}

Return ONLY this JSON (no markdown, no extra text):
{{
  "objective_and_hypothesis": {{
    "score": 3,
    "justification": "brief explanation grounded in the text, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "public_health_relevance": {{
    "score": 2,
    "justification": "brief explanation grounded in the text, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "study_design_rigor": {{
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
    "off_topic_references": ["13 - Chen et al., fungal genome assembly, unrelated to this paper's subject"],
    "non_peer_reviewed_count": 3,
    "self_citation_count": 2,
    "justification": "brief explanation of reference quality and coverage, naming any off-topic entries",
    "quote": "sample reference from the paper"
  }}
}}"""


def _build_content_prompt_pdf(criteria: dict) -> str:
    """Same rubric/procedure as _build_content_prompt, but for the attached-PDF
    path: no full_text is embedded — the model reads the PDF file directly.
    Keep in sync with _build_content_prompt if the rubric wording changes;
    _build_content_prompt itself must not change (evaluate_preprint_claude.py
    still uses it verbatim for the text-only Claude path)."""
    rqm = criteria["research_question_and_methods"]
    sub = rqm["sub_criteria"]
    obj_c = sub["objective_and_hypothesis"]
    phr_c = sub["public_health_relevance"]
    dr_c = sub["study_design_rigor"]
    rac = criteria["results_and_conclusion"]
    ref = criteria["references"]

    return f"""You are evaluating a scientific preprint for GISAID eligibility.
Read the attached PDF in full — including its figures and tables — and score
the criteria below using ONLY the evidence present.

=== CRITERION 1: {rqm["label"]} — 3 INDEPENDENT SUB-CRITERIA ===
Score each sub-criterion on its own evidence. A weak result on one must NOT pull
down your score on another.

--- Sub-criterion 1a: {obj_c["label"]} ---
Score 1: {obj_c["score_1"]["description"]}
Score 2: {obj_c["score_2"]["description"]}
Score 3: {obj_c["score_3"]["description"]}

--- Sub-criterion 1b: {phr_c["label"]} ---
Score 1: {phr_c["score_1"]["description"]}
Score 2: {phr_c["score_2"]["description"]}
Score 3: {phr_c["score_3"]["description"]}

--- Sub-criterion 1c: {dr_c["label"]} ---
Score 1: {dr_c["score_1"]["description"]}
Score 2: {dr_c["score_2"]["description"]}
Score 3: {dr_c["score_3"]["description"]}

=== CRITERION 2: {rac["label"]} ===
Score 1: {rac["score_1"]["description"]}
Score 2: {rac["score_2"]["description"]}
Score 3: {rac["score_3"]["description"]}

=== CRITERION 3: {ref["label"]} ===
Score 1: {ref["score_1"]["description"]}
Score 2: {ref["score_2"]["description"]}
Score 3: {ref["score_3"]["description"]}

=== HOW TO SCORE EACH (SUB-)CRITERION ===
For each of the 5 things you are scoring (3 sub-criteria of criterion 1, plus
criteria 2 and 3), follow this procedure before committing to a score:
1. Check the score_3 description first. Does the paper clearly meet it? If yes, score 3 — stop.
2. If not, check the score_1 description. Does the paper clearly show those weaknesses? If yes, score 1 — stop.
3. Only score 2 if the paper genuinely sits between the two — some strong elements
   present, but not enough for a clear 3, with no red flags severe enough for a clear 1.

Score 2 is not a safe default. It should be your conclusion in a minority of cases,
not most of them. Base "lacks supporting figures or tables" strictly on whether the
PDF actually contains them — check the real document, don't assume.

- Quote must be a verbatim phrase copied from the paper (max 220 characters), one
  per sub-criterion/criterion.
- Justify in 1-2 sentences, naming which score_1/score_2/score_3 description the
  evidence matches. When a figure or table is central to your reasoning, describe
  what it shows in the justification (quotes can only be text, not images).
- For results_and_conclusion: actively cross-check numeric values across tables,
  figures, and the surrounding text for internal consistency — do not just check
  whether tables/figures are present. Specifically: (1) flag any count of discrete
  items (reads, sequences, samples) reported with a decimal/fraction — that is
  intrinsically impossible and signals a data-integrity problem, not merely
  "unclear parameters"; (2) check whether a summary statistic stated in the text
  (a mean, a total) matches the values in the table it is summarizing; (3) look at
  the actual figure panels (not just their captions) for implausible patterns —
  e.g. a count of exactly zero in every sample for something the method should
  sometimes detect — which can signal a pipeline artifact rather than a real
  finding; (4) if a specific data point directly contradicts a conclusion the
  paper itself states (e.g., the paper claims genome integrity/concordance "was
  well preserved" but its own reported identity numbers show a large drop for one
  sample), that is direct evidence for score_1's "conclusions ... not grounded in
  the evidence presented" — do not soften this to score_2's "unclear statistical
  parameters", which is for vague or incomplete reporting, not a documented
  contradiction between the paper's own data and its own conclusion.
- For references: report facts, not a score. The score is computed in code from what
  you report, so report each field carefully and do not try to reason about the final
  number.
  (a) "reference_count": the total number of entries in the reference list, as an
  integer. Count them; do not estimate.
  (b) "off_topic_references": a list of the entries that are genuinely unrelated to
  this paper's subject matter (e.g. a paper about virus X citing a genome-assembly
  study of an unrelated fungus, or a plant-virology review, with no clear
  methodological link). Identify each by number and first author, with a few words
  on why. Return an empty list if there are none — an empty list is the expected
  answer for a well-focused bibliography, so do not manufacture entries, but do not
  overlook a real one either: a single off-topic entry caps this criterion below the
  top score.
  (c) "non_peer_reviewed_count": how many entries are preprints, blog posts, press
  releases or other non-peer-reviewed sources, as an integer.
  (d) "self_citation_count": how many entries are by this paper's own authors, as an
  integer.
  Write the justification about source quality and coverage (peer-reviewed vs. not,
  foundational vs. recent balance, domain relevance, self-citation reliance), naming
  the off-topic entries you found. Do not write about whether the count itself feels
  low or high, since that judgment is not used.

Return ONLY this JSON (no markdown, no extra text):
{{
  "objective_and_hypothesis": {{
    "score": 3,
    "justification": "brief explanation grounded in the paper, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "public_health_relevance": {{
    "score": 2,
    "justification": "brief explanation grounded in the paper, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "study_design_rigor": {{
    "score": 3,
    "justification": "brief explanation grounded in the paper, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "results_and_conclusion": {{
    "score": 1,
    "justification": "brief explanation grounded in the paper, naming the matched score description",
    "quote": "exact verbatim quote from the paper"
  }},
  "references": {{
    "reference_count": 24,
    "off_topic_references": ["13 - Chen et al., fungal genome assembly, unrelated to this paper's subject"],
    "non_peer_reviewed_count": 3,
    "self_citation_count": 2,
    "justification": "brief explanation of reference quality and coverage, naming any off-topic entries",
    "quote": "sample reference from the paper"
  }}
}}"""


def _build_author_prompt(author_data: dict, criteria: dict) -> str:
    extracted = author_data["extracted"]
    ror = author_data["ror_results"]
    institution_source = author_data.get("institution_source", "ror_fallback")
    s2 = author_data.get("semantic_scholar", {})
    s2_authors = s2.get("authors", [])
    all_warnings = author_data.get("data_quality_warnings", [])

    sub = criteria["author_credibility"]["sub_criteria"]
    inst_c = sub["institution_reputability"]
    exp_c = sub["author_expertise"]
    collab_c = sub["institutional_collaboration"]

    paper_fields = author_data.get("paper_fields", [])

    ror_summary = json.dumps(ror, indent=2) if ror else "No institution data found"
    s2_summary = json.dumps(s2_authors, indent=2) if s2_authors else "No Semantic Scholar data found"
    fields_summary = ", ".join(paper_fields) if paper_fields else "Unknown (no OpenAlex field data for this paper)"
    warnings_block = "\n".join(f"- {w}" for w in all_warnings) if all_warnings else "None"

    return f"""You are evaluating the authors and institutions of a scientific preprint for GISAID eligibility.
Author credibility is scored as 3 INDEPENDENT sub-criteria — score each one on its
own evidence. Do not let a weak result on one pull down your score on another.

=== SUB-CRITERION 1: {inst_c["label"]} ===
Score 1: {inst_c["score_1"]["description"]}
Score 2: {inst_c["score_2"]["description"]}
Score 3: {inst_c["score_3"]["description"]}

=== SUB-CRITERION 2: {exp_c["label"]} ===
Score 1: {exp_c["score_1"]["description"]}
Score 2: {exp_c["score_2"]["description"]}
Score 3: {exp_c["score_3"]["description"]}

=== SUB-CRITERION 3: {collab_c["label"]} ===
Score 1: {collab_c["score_1"]["description"]}
Score 2: {collab_c["score_2"]["description"]}
Score 3: {collab_c["score_3"]["description"]}

=== EXTRACTED AUTHOR DATA ===
Corresponding/first author: {extracted.get("corresponding_author", "unknown")}
All authors ({extracted.get("n_authors", "?")}): {", ".join(extracted.get("all_authors", [])[:8])}
Distinct affiliations ({extracted.get("n_institutes", "?")}):
{chr(10).join("- " + a for a in extracted.get("affiliations", []))}

=== INSTITUTION DATA (source: {institution_source}) ===
{ror_summary}

=== AUTHOR PUBLICATION STATS (Semantic Scholar, incl. "field_match" from OpenAlex) ===
This paper's own research fields: {fields_summary}
{s2_summary}

=== DATA QUALITY WARNINGS ===
{warnings_block}

=== HOW TO SCORE EACH SUB-CRITERION ===
Follow this procedure independently for each of the 3 sub-criteria above:
1. Check its score_3 description first. Does the evidence clearly meet it? If yes, score 3 — stop.
2. If not, check its score_1 description. Does the evidence clearly show those weaknesses? If yes, score 1 — stop.
3. Only score 2 if the evidence genuinely sits between the two.

Score 2 is not a safe default — it should be your conclusion in a minority of cases,
not most of them. A weak result on one sub-criterion must NOT pull down another —
e.g. a paper with only 3 institutions still scores independently on institution
reputability and author expertise; it isn't capped just because sub-criterion 3
(institutional_collaboration) tops out at 2.

=== INSTRUCTIONS ===
- Sub-criterion 1 (institution reputability): use the "type" field to judge credibility —
  "education"/"Education" or "facility"/"Research" institutions are credible; "company"/"Company"
  or missing data is a flag. Same vocabulary whether the source is OpenAlex or ROR. For score_3,
  use each institution's own "position" field (already "first"/"middle"/"last", not something to
  infer by matching names against the corresponding_author string above) to identify the
  main/corresponding author's institution directly — "first" or "last" is the main/corresponding
  author by academic convention.
- Sub-criterion 2 (author expertise): use the publication stats (paper_count > 10 and h_index > 3
  suggests an established researcher). The author list above may cover all authors on the paper or
  only some — check DATA QUALITY WARNINGS to know which. "match_method": "doi" or "openalex_fallback"
  both mean the record was matched via the paper's own author list on that source (reliable, treat
  the same way); "name_search_unverified" means it was matched by name alone and could belong to a
  different person with the same name — treat it as a weak signal.
  The rubric asks for expertise "in the field" — a high paper_count/h_index is NOT enough on its own.
  Use "field_match": true means that author's own top research fields overlap with this paper's fields
  above — count them toward "authors with verifiable expertise". "field_match": false means their
  record, however large, doesn't evidently connect to this paper's field — do NOT count them, even if
  paper_count/h_index look strong (real case: a co-author's OpenAlex profile turned out to be 141
  papers, 100% dentistry, for a virology paper — high volume, wrong field). "field_match": null means
  no field data was available — treat as unverified, not as a negative signal.
  Weight "position" when judging overall strength: "first" and "last" authors carry more weight than
  "middle" — by academic convention the first author is typically the main contributor and the last
  author is typically the senior/corresponding researcher vouching for the work. A paper where the
  first or last author has field_match:true and strong stats is stronger evidence than the same signal
  on a middle author; conversely, a first/last author with field_match:false or no verifiable expertise
  is a more significant gap than the same being true of one middle author among many.
- Sub-criterion 3 (institutional collaboration): count distinct institutions from INSTITUTION DATA above.
- If ROR/OpenAlex or Semantic Scholar returned no data, or DATA QUALITY WARNINGS above flags
  missing/incomplete coverage, note the uncertainty but do not automatically penalize — unverified
  is not the same as lacking credibility or expertise.

Return ONLY this JSON (no markdown, no extra text):
{{
  "institution_reputability": {{
    "score": 2,
    "justification": "brief explanation"
  }},
  "author_expertise": {{
    "score": 2,
    "justification": "brief explanation"
  }},
  "institutional_collaboration": {{
    "score": 2,
    "justification": "brief explanation"
  }},
  "institutions_assessed": ["list of institution names"],
  "n_institutes": 2,
  "corresponding_author_papers": 0,
  "flags": ["any concerns found, e.g. commercial affiliation"]
}}"""


def _score_from_reference_count(n: int) -> int:
    """Deterministic score from a reference count. The LLM only has to count,
    not compare against the threshold (it was getting the comparison wrong,
    e.g. calling 58 'within 10-20')."""
    if n > 20:
        return 3
    if n < 10:
        return 1
    return 2


# Sandy's rubric asks for a "balanced mix of foundational, recent and domain-relevant"
# peer-reviewed work for a 3, and calls "overreliance on non-peer-reviewed, obscure or
# self-citations" a 1. Scoring purely on the count answered neither: one calibration
# paper cited a fungal genome assembly and a fly-sequencing study, the model said so in
# its own justification, and still scored 3 because it had 33 entries. So the count sets
# the base and the quality findings can only pull it down, never push it up. The model
# reports what it found; the rule lives here, same split as the count itself.
_OFF_TOPIC_CAP = 2       # any genuinely off-topic entry blocks the "domain-relevant" requirement of a 3
_LOW_QUALITY_SHARE = 0.5  # provisional, like the _recommendation thresholds: calibrate against labelled cases


def _apply_reference_score(ref: dict) -> dict:
    """Set references' score from the count, then apply the quality gates.
    Fields other than reference_count are optional: results produced before
    they existed score exactly as they did then."""
    n = ref.get("reference_count")
    if not isinstance(n, int):
        ref["score"] = 1
        ref["justification"] = (ref.get("justification", "") + " [no reference_count reported]").strip()
        return ref

    score = _score_from_reference_count(n)
    notes = []

    off_topic = ref.get("off_topic_references") or []
    if off_topic and score > _OFF_TOPIC_CAP:
        score = _OFF_TOPIC_CAP
        notes.append(f"capped at {_OFF_TOPIC_CAP}: {len(off_topic)} off-topic reference(s) reported "
                     f"({'; '.join(str(x) for x in off_topic)[:200]})")

    low_quality = (ref.get("non_peer_reviewed_count") or 0) + (ref.get("self_citation_count") or 0)
    if n > 0 and low_quality / n > _LOW_QUALITY_SHARE and score > 1:
        score = 1
        notes.append(f"dropped to 1: {low_quality} of {n} references are non-peer-reviewed "
                     f"or self-citations, over the {_LOW_QUALITY_SHARE:.0%} overreliance threshold")

    ref["score"] = score
    if notes:
        ref["score_note"] = " | ".join(notes)
    return ref


# ─── reference-list citation verification (Crossref, no LLM) ────────────────
# Added 2026-07-24 after a real calibration paper's references list turned out
# to include several entries genuinely unrelated to its subject (see criteria.yaml
# references.description) while the LLM's own justification called the list
# "domain-relevant". This is a separate, purely deterministic check: does each
# printed citation actually resolve to a real indexed work, via Crossref (free,
# no API key). It does not judge topical relevance — it catches garbled/
# fabricated/mismatched citations, the same "verify against a real external
# source instead of trusting the LLM's read" principle already used for
# institutions (ROR/OpenAlex) and authors (Semantic Scholar) elsewhere in this
# file.

_DOI_RE = re.compile(
    r'10\.\d{4,9}/[-._;/:a-zA-Z0-9]+(?:\([-._;/:a-zA-Z0-9]+\)[-._;/:a-zA-Z0-9]*)*'
)  # DOI suffixes legitimately contain balanced parentheses (e.g. Elsevier/Lancet-
   # style "10.1016/S2542-5196(20)30178-9") — a naive "stop at any ')'" regex
   # truncates those mid-DOI; this allows matched "(...)" groups through instead

_REF_NAME_PREFIX = r'(?:de|van|von|da|dos|del|la|le)\s+'
_REF_SURNAME = r"[A-ZÀ-Ý][a-zà-ÿ']+(?:-[A-ZÀ-Ý][a-zà-ÿ']+)*"
_REF_ENTRY_RE = re.compile(
    rf'(?:^|\n)\s*(?:Page\s+\d+/\d+\s*\n\s*)?\d{{0,3}}\s*\.\s+'
    rf'(?=(?:{_REF_NAME_PREFIX})?{_REF_SURNAME}(?:\s+[A-ZÀ-Ý]|,))'
)

# Kept for reference in the stored output only. It used to decide acceptance, and
# rejected a word-for-word correct match at 48 ("The global distribution and burden
# of dengue"): Crossref's score is not normalised, so no fixed cutoff works.
# _match_is_plausible now makes the call from the record itself.
_CROSSREF_MATCH_SCORE_THRESHOLD = 50


def _clean_doi(raw: str) -> str:
    return raw.rstrip(".").rstrip(")").rstrip("]")


# The heading is matched on its own line and case-insensitively: one paper in the
# calibration set writes "REFERENCES", and a case-sensitive rfind("References")
# silently dropped its whole 67-entry list before any parsing began.
_REF_HEADING_RE = re.compile(
    r"(?im)^[^\S\n]{0,12}(?:references|bibliography|literature cited|works cited)"
    r"[^\S\n]*:?[^\S\n]*$")

# Line numbers down the margin of a preprint come out of pdfminer as their own
# block ("1\n\n2\n\n3..."), and page furniture lands inside entries.
_LINE_NUMBER_BLOCK_RE = re.compile(r"(?:^|\n)(?:[^\S\n]*\d{1,4}[.)]?[^\S\n]*\n\s*){3,}")
_PAGE_FURNITURE_RE = re.compile(r"(?im)^[^\S\n]*Page\s+\d+\s*/\s*\d+[^\S\n]*$\n?")


def _extract_references_section(full_text: str) -> str:
    """Isolate the reference-list text from the end of the paper. Used only
    for citation verification (verify_references), independent of whatever
    slice of the paper is sent to the LLM for scoring."""
    matches = list(_REF_HEADING_RE.finditer(full_text))
    if matches:
        section = full_text[matches[-1].end():]
    else:
        # No heading on a line of its own: fall back to the last mention of the
        # word anywhere, case-insensitively.
        low = full_text.lower()
        idx = low.rfind("references")
        if idx == -1:
            return ""
        section = full_text[idx + len("references"):]

    for stop in ("\nFigures\n", "\nSupplementary", "\nFigure Legends", "\nAcknowledg"):
        cut = section.lower().find(stop.lower())
        if cut != -1:
            section = section[:cut]

    section = _PAGE_FURNITURE_RE.sub("", section)
    section = _LINE_NUMBER_BLOCK_RE.sub("\n", section)
    return section


# Three reference-list styles seen in practice. The numbered ones are tried first
# and keep their existing behaviour; the unnumbered strategy exists because a
# line-numbered preprint loses its entry numbers to pdfminer's margin block, which
# left one paper's 44 entries parsing as zero.
_REF_ENTRY_BRACKET_RE = re.compile(rf'(?:^|\n)\s*\[\d{{1,3}}\]\s*'
                                   rf'(?=(?:{_REF_NAME_PREFIX})?{_REF_SURNAME}(?:\s+[A-ZÀ-Ý]|,))')
_REF_ENTRY_UNNUMBERED_RE = re.compile(rf'(?:^|\n)\s*'
                                      rf'(?=(?:{_REF_NAME_PREFIX})?{_REF_SURNAME}(?:\s+[A-ZÀ-Ý]\b|,))')
# "27 Li, M. et al." — numbered, but with no period after the number.
_REF_ENTRY_BARE_NUMBER_RE = re.compile(rf'(?:^|\n)\s*\d{{1,3}}[.)]?\s+'
                                       rf'(?=(?:{_REF_NAME_PREFIX})?{_REF_SURNAME}(?:\s+[A-ZÀ-Ý]|,))')


def _split_on(pattern: re.Pattern, ref_section: str) -> list[str]:
    positions = [m.start() for m in pattern.finditer(ref_section)]
    entries = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(ref_section)
        entry = re.sub(r"\s+", " ", ref_section[pos:end]).strip()
        if entry:
            entries.append(entry)
    return entries


def _split_reference_entries(ref_section: str) -> list[str]:
    """Split a references section into individual entries, tolerant of real
    extraction quirks found in practice: pdfminer drops the odd leading digit
    in some reference numbers (e.g. "16." extracts as "1 ."), and entries
    commonly start with a hyphenated compound surname ("Cardona-Trujillo") or
    a lowercase surname prefix ("de Souza") — plain "^\\d+\\. " splitting misses
    both.

    Every strategy is tried and the one that finds the most entries wins. That
    sounds like it should over-split, but measured across the calibration set it
    does not: the winning strategy never exceeded the paper's real reference
    count by more than five, while first-past-the-post selection left four
    papers parsing under 11% of their list because a numbered pattern matched a
    handful of false positives and blocked the unnumbered strategy that would
    have found the whole list.

    Known residual gap (verified, not fixed): a Chinese-name romanization with
    the hyphen inside a lowercase given name (e.g. "Wei-ying Chen") isn't caught
    — that entry merges into its neighbour's text instead of being dropped, so
    verification runs on the combined text rather than skipping a reference."""
    candidates = [_split_on(pattern, ref_section) for pattern in
                  (_REF_ENTRY_RE, _REF_ENTRY_BRACKET_RE,
                   _REF_ENTRY_BARE_NUMBER_RE, _REF_ENTRY_UNNUMBERED_RE)]
    return max(candidates, key=len)


def _crossref_get(url: str, params: dict, session: requests.Session, max_retries: int = 3) -> requests.Response | None:
    """GET with retry-with-backoff on 429 specifically — Crossref's anonymous
    rate limit is 1 req/s (3 req/s with CROSSREF_MAILTO's "polite pool"), and a
    verification pass hits it dozens of times per paper in quick succession.
    Without this, a rate-limited request looks identical to a genuine
    no-match — verified in practice: the same query flipped between a
    confident match and total failure across repeated runs purely because of
    intermittent 429s, which would have silently mislabeled real citations as
    unverifiable. Only retries 429; any other failure (timeout, 5xx, network
    error) still returns None immediately, same as before."""
    params = {**params, "mailto": CROSSREF_MAILTO}
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=10)
        except Exception:
            return None
        if r.status_code == 429:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        return r
    return None


_TITLE_STOPWORDS = frozenset(
    "a an the of in and for to on with by from as at is are its their this that "
    "we be been between during via using use over under into after before".split())
_TITLE_OVERLAP_THRESHOLD = 0.6  # share of the matched title's content words that must appear in the citation


def _crossref_record(item: dict) -> dict:
    """The fields worth keeping from a Crossref work. type distinguishes a
    preprint ("posted-content") from a published article ("journal-article"),
    and the author list is what a self-citation check needs; both used to be
    thrown away here and re-derived from the model's reading instead."""
    titles = item.get("title", [])
    issued = (item.get("issued", {}) or {}).get("date-parts", [[]])
    year = issued[0][0] if issued and issued[0] else None
    return {
        "doi": item.get("DOI", ""),
        "title": titles[0] if titles else "",
        "type": item.get("type", ""),
        "container": (item.get("container-title") or [""])[0],
        "authors": [a.get("family", "") for a in (item.get("author") or []) if a.get("family")],
        "year": year,
        "cited_by": item.get("is-referenced-by-count"),
    }


def _title_overlap(title: str, entry_text: str) -> float:
    """Share of the title's content words that actually appear in the citation."""
    words = [w for w in re.findall(r"[a-z0-9]+", _norm(title)) if w not in _TITLE_STOPWORDS and len(w) > 2]
    if not words:
        return 0.0
    entry_words = set(re.findall(r"[a-z0-9]+", _norm(entry_text)))
    return sum(1 for w in words if w in entry_words) / len(words)


def _match_is_plausible(entry_text: str, rec: dict) -> tuple[bool, str]:
    """Decide whether a Crossref hit really is the work the entry cites.

    Crossref's own relevance score is not normalised and makes a poor cutoff:
    at the previous threshold of 50, "The global distribution and burden of
    dengue" was rejected at 48 against a citation whose title matched it word
    for word. So the decision is made from the record itself instead, the same
    way quotes are checked against the paper text rather than trusted."""
    overlap = _title_overlap(rec.get("title", ""), entry_text)
    if overlap < _TITLE_OVERLAP_THRESHOLD:
        return False, f"title overlap {overlap:.0%} below {_TITLE_OVERLAP_THRESHOLD:.0%}"

    entry_norm = _norm(entry_text)
    first_author = (rec.get("authors") or [""])[0]
    author_ok = bool(first_author) and _norm(first_author) in entry_norm
    year_ok = bool(rec.get("year")) and str(rec["year"]) in entry_text
    if not (author_ok or year_ok):
        return False, f"title overlap {overlap:.0%} but neither first author nor year appears in the entry"
    corroboration = "author" if author_ok else "year"
    return True, f"title overlap {overlap:.0%}, corroborated by {corroboration}"


def _crossref_lookup_doi(doi: str, session: requests.Session) -> dict:
    """Verify a DOI already printed in the reference actually resolves."""
    r = _crossref_get(f"{CROSSREF_API}/{doi}", {}, session)
    if r is None or r.status_code != 200:
        return {}
    try:
        item = r.json().get("message", {})
    except Exception:
        return {}
    return _crossref_record(item)


def _crossref_search_bibliographic(entry_text: str, session: requests.Session) -> dict:
    """No DOI printed in the entry — search Crossref's own bibliographic-string
    matcher for the best candidate. The ranking (and the accept/reject call
    via _CROSSREF_MATCH_SCORE_THRESHOLD) is Crossref's deterministic score,
    not an LLM judgment."""
    r = _crossref_get(CROSSREF_API, {"query.bibliographic": entry_text, "rows": 1}, session)
    if r is None or r.status_code != 200:
        return {}
    try:
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return {}
    if not items:
        return {}
    best = items[0]
    rec = _crossref_record(best)
    rec["score"] = best.get("score", 0)
    return rec


def verify_references(full_text: str, session: requests.Session | None = None) -> dict:
    """Non-LLM verification of the reference list via Crossref: confirms each
    entry either carries a DOI that actually resolves, or can be matched to a
    real indexed work by bibliographic search. Purely informational — does
    NOT affect the deterministic reference_count score (_score_from_reference_count
    stays the only thing driving the score, by design). Flags entries that
    can't be confirmed as a real, indexed publication — can indicate a garbled
    extraction, a fabricated/mismatched citation, OR simply a work outside
    Crossref's coverage (some gray literature, government/agency reports,
    non-English-indexed journals) — "unresolved" is a prompt to check
    manually, not proof of fabrication."""
    session = session or requests.Session()
    section = _extract_references_section(full_text)
    entries = _split_reference_entries(section)

    checked = []
    for i, entry in enumerate(entries):
        if i > 0:
            time.sleep(0.35)  # stay under the polite pool's 3 req/s even before any 429 forces a retry
        doi_match = _DOI_RE.search(entry)
        resolved_via_doi = False
        if doi_match:
            doi = _clean_doi(doi_match.group(0))
            result = _crossref_lookup_doi(doi, session)
            if result:
                checked.append({"entry": entry[:160], "resolved": True,
                                 "method": "doi_in_text", "doi": result["doi"],
                                 "record": result})
                resolved_via_doi = True

        if not resolved_via_doi:
            # Either no DOI was printed, or the one that was didn't resolve —
            # the latter is often not a bad citation but a text-extraction
            # artifact: verified in practice that this PDF's extraction drops
            # hyphens where a DOI wraps across a line break (e.g.
            # "S0065-3527" → "S00653527"), which silently breaks an exact-DOI
            # lookup for an otherwise perfectly real, resolvable reference.
            # Bibliographic search on the full entry text doesn't depend on
            # getting that exact string right, so try it as a fallback before
            # concluding the citation can't be confirmed.
            result = _crossref_search_bibliographic(entry, session)
            plausible, why = _match_is_plausible(entry, result) if result else (False, "no candidate returned")
            if plausible:
                checked.append({"entry": entry[:160], "resolved": True,
                                 "method": "bibliographic_search" if not doi_match else "bibliographic_search_doi_fallback",
                                 "doi": result["doi"], "match_score": result.get("score"),
                                 "match_check": why, "record": result})
            else:
                checked.append({"entry": entry[:160], "resolved": False,
                                 "method": "bibliographic_search_no_match" if not doi_match else "doi_in_text_not_resolved",
                                 "match_check": why,
                                 **({"doi": _clean_doi(doi_match.group(0))} if doi_match else {})})

    n_resolved = sum(1 for c in checked if c["resolved"])
    return {
        "n_entries_found": len(checked),
        "n_resolved": n_resolved,
        "n_unresolved": len(checked) - n_resolved,
        "unresolved": [c for c in checked if not c["resolved"]],
    }


# ─── PREreview feedback lookup (no LLM for fetching; real HTML, no JS) ──────
# Added 2026-07-24 (feedback criterion review). PREreview.org publishes expert
# peer reviews of preprints at a predictable URL, server-rendered — no API
# key, no JavaScript rendering needed (verified against a real reviewed
# preprint and a real unreviewed one). This is the only feedback-criterion
# source found with a clean, scrapeable signal — bioRxiv/medRxiv's own
# Disqus-based comments remain JS-only and out of scope. Two things this
# deliberately does NOT do (per Juan's correction, 2026-07-24): does not use
# "already published in a journal" as a feedback signal (that's a workflow
# question — should this even still be in the significant-preprint queue —
# not evidence of community feedback), and does not use citation counts (that
# measures broader scientific impact, not direct feedback/discussion on the
# preprint itself, which is what this criterion actually asks about).

def _prereview_doi_slug(doi: str) -> str:
    return "doi-" + doi.replace("/", "-")


def _extract_prereview_body(stripped_text: str) -> str:
    """Crop the page's chrome (nav/menu/footer) from its stripped text. Not
    surgically precise — some header metadata (title/author/date/license)
    can leak into the start — but that's harmless for an LLM reader; the
    goal is just to exclude the site-wide nav and footer boilerplate."""
    start_marker = "Read the preprint"
    end_marker = "Learn about upcoming events"
    start = stripped_text.find(start_marker)
    start = start + len(start_marker) if start != -1 else 0
    end = stripped_text.find(end_marker)
    if end == -1 or end <= start:
        end = len(stripped_text)
    return stripped_text[start:end].strip()


def _strip_html_text(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def fetch_prereview_data(doi: str, session: requests.Session | None = None) -> dict:
    """Fetch PREreview.org's page for a preprint DOI and, if any reviews
    exist, pull the full text of each one. Returns
    {"n_reviews": int, "reviews": [{"review_id", "text"}, ...]}. Any failure
    (network error, DOI not found on PREreview) returns n_reviews=0 — treated
    the same as "genuinely zero reviews" by evaluate_feedback, since either
    way there's no automated evidence to score from."""
    session = session or requests.Session()
    try:
        r = session.get(f"{PREREVIEW_BASE}/preprints/{_prereview_doi_slug(doi)}", timeout=10)
        if r.status_code != 200:
            return {"n_reviews": 0, "reviews": []}
        review_ids = sorted(set(re.findall(r"/reviews/(\d+)", r.text)))
    except Exception:
        return {"n_reviews": 0, "reviews": []}

    reviews = []
    for rid in review_ids:
        try:
            rr = session.get(f"{PREREVIEW_BASE}/reviews/{rid}", timeout=10)
            if rr.status_code != 200:
                continue
            body = _extract_prereview_body(_strip_html_text(rr.text))
            if body:
                reviews.append({"review_id": rid, "text": body})
        except Exception:
            continue
    return {"n_reviews": len(reviews), "reviews": reviews}


def _build_feedback_prompt(criteria: dict, review_texts: list[str]) -> str:
    fb = criteria["feedback"]
    reviews_block = "\n\n".join(f"=== PREreview #{i + 1} ===\n{t}" for i, t in enumerate(review_texts))

    return f"""You are evaluating community feedback on a scientific preprint for GISAID eligibility.
You have been given the full text of {len(review_texts)} public PREreview(s) — independent expert
peer review(s) published on prereview.org for this specific preprint. Score using ONLY this evidence.

=== CRITERION: {fb["label"]} ===
Score 1: {fb["score_1"]["description"]}
Score 2: {fb["score_2"]["description"]}
Score 3: {fb["score_3"]["description"]}

=== IMPORTANT CAVEAT ===
This evidence covers PREreview only. It does NOT include comments on the preprint's own
hosting server (bioRxiv/medRxiv/ResearchSquare), which are not available to you and were not
checked. Score strictly from what these PREreview(s) show — their depth, expertise, and
whether they read as a full expert review versus a cursory comment. Explicitly note in the
justification that server-side comments were not checked, rather than assuming anything
about them.

{reviews_block}

Return ONLY this JSON (no markdown, no extra text):
{{
  "score": 2,
  "justification": "brief explanation grounded in the review text, naming which score description it matches, and noting that bioRxiv/medRxiv comments were not checked"
}}"""


def evaluate_feedback(doi: str, criteria: dict, client: OpenAI, model: str,
                       session: requests.Session | None = None) -> dict:
    """Score the feedback criterion from real PREreview text when available.
    When there are zero PREreviews for this DOI, stays fully manual
    (score: None) — absence of a PREreview does NOT mean absence of feedback,
    since bioRxiv/medRxiv server-side comments might still exist unseen."""
    prereview_data = fetch_prereview_data(doi, session)
    if prereview_data["n_reviews"] == 0:
        return {
            "score": None,
            "justification": "Not evaluated automatically. Requires manual check of preprint server.",
            "prereview_data": prereview_data,
        }

    prompt = _build_feedback_prompt(criteria, [rv["text"] for rv in prereview_data["reviews"]])
    raw = _llm(client, model, prompt, max_tokens=2000, reasoning_effort="medium")
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "score": None,
            "justification": "Parse error scoring PREreview evidence — requires manual check.",
            "prereview_data": prereview_data,
        }

    score = parsed.get("score")
    return {
        "score": score if isinstance(score, int) else None,
        "justification": parsed.get("justification", ""),
        "prereview_data": prereview_data,
    }


_RESEARCH_SUB_CRITERIA = ("objective_and_hypothesis", "public_health_relevance", "study_design_rigor")


def _compose_research_question_result(parsed: dict) -> dict:
    """Combine the 3 independent research_question_and_methods sub-scores into
    the final composite (average, rounded to the nearest integer) — same
    principle as evaluate_author_credibility's composite and
    _score_from_reference_count: the aggregate is computed in code, never left
    to the LLM's own arithmetic. Each sub-score keeps its own quote so
    validate_quotes can check/downgrade them independently."""
    sub_scores = {}
    for key in _RESEARCH_SUB_CRITERIA:
        block = parsed.get(key, {})
        score = block.get("score")
        sub_scores[key] = {
            "score": score if isinstance(score, int) else 1,
            "justification": block.get("justification", ""),
            "quote": block.get("quote", ""),
        }

    composite = round(sum(s["score"] for s in sub_scores.values()) / len(_RESEARCH_SUB_CRITERIA))
    combined_justification = " | ".join(
        f"{key}: {sub_scores[key]['justification']}" for key in _RESEARCH_SUB_CRITERIA
    )

    return {
        "score": composite,
        "justification": combined_justification,
        "sub_scores": sub_scores,
    }


_RAW_SNIPPET_CAP = 4000


def _failure_payload(reason: str, raw: str = "", finish_reason: str | None = None,
                     usage: object = None) -> dict:
    """Diagnostics for a call that never produced a usable answer. Without the
    raw response on disk there is no way to tell a truncated reply from a
    refusal or a malformed one after the fact."""
    payload = {"reason": reason}
    if finish_reason:
        payload["finish_reason"] = finish_reason
    if usage is not None:
        tokens = {"prompt": getattr(usage, "prompt_tokens", None),
                  "completion": getattr(usage, "completion_tokens", None)}
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            tokens["reasoning"] = getattr(details, "reasoning_tokens", None)
        payload["tokens"] = tokens
    if raw:
        payload["raw_response"] = raw[:_RAW_SNIPPET_CAP]
        payload["raw_response_chars"] = len(raw)
    return payload


def _failed_content_result(reason: str, raw: str = "", finish_reason: str | None = None,
                           usage: object = None) -> dict:
    """Criteria 2-4 when the content call produced nothing parseable.

    Scores stay None rather than 1: a failure written as the lowest score
    aggregates into a perfectly plausible "reject" and is indistinguishable
    from a real one. compute_scores turns any of these into an "error"
    recommendation instead of averaging them.
    """
    err = _failure_payload(reason, raw, finish_reason, usage)
    justification = f"Evaluation failed: {reason}"
    return {
        "research_question_and_methods": {
            "score": None,
            "justification": justification,
            "sub_scores": {},
            "error": err,
        },
        "results_and_conclusion": {"score": None, "justification": justification,
                                   "quote": "", "error": err},
        "references": {"score": None, "justification": justification,
                       "quote": "", "error": err},
    }


def evaluate_content(full_text: str, criteria: dict, client: OpenAI, model: str) -> dict:
    """Evaluate criteria 2, 3, 4 (research question, results, references) from full paper text."""
    prompt = _build_content_prompt(full_text, criteria)
    raw, finish_reason, usage = _llm_call(
        client, model, [{"role": "user", "content": prompt}], _CONTENT_MAX_TOKENS)
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    if not raw:
        return _failed_content_result("empty response from model", raw, finish_reason, usage)
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_content_result(f"could not parse response as JSON ({e})",
                                      raw, finish_reason, usage)

    result = {
        "research_question_and_methods": _compose_research_question_result(parsed),
        "results_and_conclusion": parsed.get("results_and_conclusion", {}),
        "references": parsed.get("references", {}),
    }

    _apply_reference_score(result["references"])
    return result


def evaluate_content_pdf(pdf_path: Path, criteria: dict, client: OpenAI, model: str) -> dict:
    """Evaluate criteria 2, 3, 4 by sending the PDF file directly (text + page
    images), instead of pre-extracted text. Requires a vision-capable model."""
    pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")
    prompt = _build_content_prompt_pdf(criteria)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "filename": pdf_path.name,
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                },
            },
            {"type": "text", "text": prompt},
        ],
    }]
    raw, finish_reason, usage = _llm_call(
        client, model, messages, _CONTENT_MAX_TOKENS, reasoning_effort="high")
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    if not raw:
        return _failed_content_result("empty response from model", raw, finish_reason, usage)
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_content_result(f"could not parse response as JSON ({e})",
                                      raw, finish_reason, usage)

    result = {
        "research_question_and_methods": _compose_research_question_result(parsed),
        "results_and_conclusion": parsed.get("results_and_conclusion", {}),
        "references": parsed.get("references", {}),
    }

    _apply_reference_score(result["references"])
    return result


_AUTHOR_SUB_CRITERIA = ("institution_reputability", "author_expertise", "institutional_collaboration")


def _compose_author_credibility_result(parsed: dict) -> dict:
    """Combine the 3 independent author_credibility sub-scores into the final
    composite (average, rounded to the nearest integer) — the composite is
    never left to the LLM's own arithmetic, same principle as
    _score_from_reference_count. Shared by both the GPT path
    (evaluate_author_credibility) and the Claude path
    (evaluate_author_credibility_claude in evaluate_preprint_claude.py) so the
    two never drift out of sync the way they did after the 2026-07-22
    sub-criteria rewrite (Claude side kept returning the raw un-composed JSON,
    with no top-level "author_credibility" key at all)."""
    sub_scores = {}
    for key in _AUTHOR_SUB_CRITERIA:
        block = parsed.get(key, {})
        score = block.get("score")
        sub_scores[key] = {
            "score": score if isinstance(score, int) else 1,
            "justification": block.get("justification", ""),
        }

    composite = round(sum(s["score"] for s in sub_scores.values()) / len(_AUTHOR_SUB_CRITERIA))
    combined_justification = " | ".join(
        f"{key}: {sub_scores[key]['justification']}" for key in _AUTHOR_SUB_CRITERIA
    )

    return {
        "author_credibility": {
            "score": composite,
            "justification": combined_justification,
            "sub_scores": sub_scores,
            "institutions_assessed": parsed.get("institutions_assessed", []),
            "n_institutes": parsed.get("n_institutes", 0),
            "corresponding_author_papers": parsed.get("corresponding_author_papers", 0),
            "flags": parsed.get("flags", []),
        }
    }


def _failed_author_credibility_result(reason: str, raw: str = "",
                                      finish_reason: str | None = None,
                                      usage: object = None) -> dict:
    """Criterion 1 when its call produced nothing parseable. Score stays None,
    same reasoning as _failed_content_result."""
    return {
        "author_credibility": {
            "score": None,
            "justification": f"Evaluation failed: {reason}",
            "sub_scores": {},
            "flags": [],
            "error": _failure_payload(reason, raw, finish_reason, usage),
        }
    }


def evaluate_author_credibility(author_data: dict, criteria: dict, client: OpenAI, model: str) -> dict:
    """Evaluate criterion 1 (author credibility) using enriched author data.

    Scores 3 independent sub-criteria (see criteria.yaml, sourced from
    tabla_de_criterios_sandy.xlsx) and combines them into the final
    author_credibility score via _compose_author_credibility_result.
    """
    prompt = _build_author_prompt(author_data, criteria)
    raw, finish_reason, usage = _llm_call(
        client, model, [{"role": "user", "content": prompt}],
        _AUTHOR_MAX_TOKENS, reasoning_effort="medium")
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    if not raw:
        return _failed_author_credibility_result("empty response from model", raw,
                                                 finish_reason, usage)
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return _failed_author_credibility_result(
            f"could not parse response as JSON ({e})", raw, finish_reason, usage)

    return _compose_author_credibility_result(parsed)


# ─── quote validation ────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    # collapse typographic variants so curly/straight quotes and dashes compare equal
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[''`]", "'", text)          # curly single quotes → '
    text = re.sub(r'[""„«»]', '"', text)        # curly double quotes → "
    text = re.sub(r"[–—−]", "-", text)          # dashes → hyphen
    # British/American institution-name spelling variants (observed real case:
    # OpenAlex matched "...Health Care Center" against a paper whose own text
    # says "...Health Care Centre" — same institution, dropped as a false mismatch)
    text = re.sub(r"\bcentre\b", "center", text)
    text = re.sub(r"\borganisation\b", "organization", text)
    text = re.sub(r"\bprogramme\b", "program", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_STANDALONE_NUMBER_RE = re.compile(r"(?<!\S)\d{1,4}(?!\S)")


def _strip_line_numbers(text: str) -> str:
    """Drop standalone 1-4 digit tokens (peer-review manuscripts embed inline
    line numbers like '...investigated \\n21\\n mosquito...' that break an
    otherwise-verbatim quote). Only strips digits with whitespace on both
    sides, so numbers glued to units/words (24%, Fig.4) survive."""
    return re.sub(r"\s+", " ", _STANDALONE_NUMBER_RE.sub(" ", text)).strip()


def _bag_of_words_match(quote_words: list[str], full_text_words: list[str], threshold: float = 0.85) -> bool:
    """Order-independent fallback: true if some window of full_text_words
    (sized generously around the quote) contains most of the quote's words.
    Catches pdfminer reading-order scrambles (multi-column layouts, callout
    boxes) that break exact and prefix matching even though the words are
    genuinely all present nearby."""
    n = len(quote_words)
    if n < 8:
        return False
    quote_set = set(quote_words)
    window = n * 2
    for i in range(0, max(len(full_text_words) - window, 0) + 1):
        if len(quote_set & set(full_text_words[i:i + window])) / len(quote_set) >= threshold:
            return True
    return False


def _quote_found(quote: str, full_text: str) -> bool:
    """Return True if quote is verifiable in full_text. Tries, in order: exact
    match, first-12-words prefix match, both again after stripping inline line
    numbers, and finally an order-independent bag-of-words match — see
    _strip_line_numbers and _bag_of_words_match for the extraction artifacts
    each step defends against."""
    q = _norm(quote)
    if not q:
        return False

    ft = _norm(full_text)
    words = q.split()

    def _prefix_hit(text: str, tokens: list[str]) -> bool:
        return len(tokens) >= 8 and " ".join(tokens[:12]) in text

    if q in ft or _prefix_hit(ft, words):
        return True

    q_nolines = _strip_line_numbers(q)
    ft_nolines = _strip_line_numbers(ft)
    words_nolines = q_nolines.split()
    if q_nolines and (q_nolines in ft_nolines or _prefix_hit(ft_nolines, words_nolines)):
        return True

    return _bag_of_words_match(words, ft.split())


def _validate_research_question_quotes(c: dict, full_text: str) -> None:
    """research_question_and_methods has 3 independently-quoted sub-scores
    (see _compose_research_question_result) instead of a single top-level
    quote — check/downgrade each sub-score on its own evidence, then
    recompute the composite score from the (possibly downgraded) sub-scores,
    same aggregation rule as _compose_research_question_result."""
    sub_scores = c.get("sub_scores", {})
    for key in _RESEARCH_SUB_CRITERIA:
        if key not in sub_scores:
            continue
        sub = sub_scores[key]
        quote = sub.get("quote", "")
        valid = _quote_found(quote, full_text) if quote else False
        sub["quote_valid"] = valid
        if not valid and sub.get("score") == 3:
            sub["score"] = 2
            sub["score_note"] = "downgraded 3→2: supporting quote not found verbatim in text"

    if sub_scores:
        c["score"] = round(sum(s["score"] for s in sub_scores.values()) / len(sub_scores))
        c["quote_valid"] = all(s.get("quote_valid", False) for s in sub_scores.values())


def validate_quotes(criterion_results: dict, full_text: str) -> dict:
    """
    For each criterion with a 'quote' field, verify the quote exists verbatim
    in full_text. Adds 'quote_valid' bool to each criterion. Downgrades score
    from 3 → 2 when the supporting quote cannot be verified.
    """
    rqm = criterion_results.get("research_question_and_methods")
    if rqm is not None and not rqm.get("error"):
        _validate_research_question_quotes(rqm, full_text)

    CRITERIA_WITH_QUOTES = [
        "results_and_conclusion",
        "references",
    ]
    for key in CRITERIA_WITH_QUOTES:
        if key not in criterion_results:
            continue
        c = criterion_results[key]
        if c.get("error"):
            continue
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
    Aggregate per-criterion scores. feedback is included when it has a real
    score (see evaluate_feedback — only happens when PREreviews exist for
    this DOI); otherwise its score stays None and it's excluded here exactly
    like before, same as any other criterion that failed to produce a score.
    Recommendation thresholds are not applied here — calibrate with real cases.
    """
    scored_criteria = ["author_credibility", "research_question_and_methods",
                       "results_and_conclusion", "references", "feedback"]

    scores = {}
    for k in scored_criteria:
        if k in criterion_results:
            scores[k] = criterion_results[k].get("score", 0)

    failed = sorted(k for k, v in criterion_results.items()
                    if isinstance(v, dict) and v.get("error"))

    valid_scores = [s for s in scores.values() if isinstance(s, int) and s > 0]
    total = sum(valid_scores)
    max_possible = len(valid_scores) * 3  # scales with how many criteria actually got a real score, not a fixed count: feedback often stays unscored
    avg = round(total / len(valid_scores), 2) if valid_scores else 0

    # A criterion that errored has no score, so it cannot be averaged into a
    # recommendation - and the remaining criteria cannot stand in for it either,
    # since scoring 1 of 4 criteria and calling the result "accept" is worse than
    # saying nothing. Any failure makes the whole evaluation an error.
    return {
        "per_criterion": scores,
        "total": total,
        "max": max_possible,
        "average": avg,
        "n_criteria_evaluated": len(valid_scores),
        "failed_criteria": failed,
        "recommendation": "error" if failed else _recommendation(avg, scores),
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

    # Step 3: evaluate content criteria (2, 3, 4) in one LLM call — PDF sent directly
    content_results = evaluate_content_pdf(pdf_path, criteria, client, model)

    # Step 4: evaluate author credibility (1) in one LLM call
    author_results = evaluate_author_credibility(author_data, criteria, client, model)

    # Merge all criterion results
    criterion_results = {**author_results, **content_results}

    # Step 4b: evaluate feedback from real PREreview text, when any exists
    # (stays manual — score: None — if this DOI has zero PREreviews)
    criterion_results["feedback"] = evaluate_feedback(_doi_from_pdf_path(pdf_path), criteria, client, model)

    # Step 5: validate quotes against full text (catches hallucinated evidence)
    criterion_results = validate_quotes(criterion_results, pdf_data["full_text"])

    # Step 5b: verify the reference list against Crossref (no LLM, informational only)
    criterion_results["references"]["citation_verification"] = verify_references(pdf_data["full_text"])

    # Step 6: aggregate scores
    scoring = compute_scores(criterion_results)

    # Add quote health summary to scoring
    checked = ["research_question_and_methods", "results_and_conclusion", "references"]
    invalid = [k for k in checked
               if not criterion_results.get(k, {}).get("error")
               and not criterion_results.get(k, {}).get("quote_valid", True)]
    scoring["quotes_invalid"] = invalid

    return {
        "source_pdf": pdf_path.name,
        "model": model,
        "n_pages": pdf_data["n_pages"],
        "author_enrichment": author_data,
        "criteria": criterion_results,
        "scoring": scoring,
        "data_quality_warnings": author_data.get("data_quality_warnings", []),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate a preprint PDF for GISAID eligibility.")
    ap.add_argument("--pdf", required=True, help="Path to the preprint PDF")
    ap.add_argument("--criteria", default="criteria.yaml", help="Path to criteria.yaml")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
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
