# Preprint Classification System

**LLM-based scoring of preprints that cite genomic data, to decide which ones are worth
linking to a genomic surveillance database.**

A preprint that reports or cites genomic sequence data is a candidate for being linked
into a genomic surveillance database. Whether it actually qualifies is a judgment about
the paper itself: who wrote it, how the study was designed, whether the results hold up,
what it cites. It has to be made one preprint at a time.

This repository automates that judgment: it takes a preprint PDF, scores it against a
five-criterion rubric using an LLM **grounded in external scholarly APIs**, validates
the model's own evidence against the source text, and emits a structured recommendation
(`accept`, `accept_with_reservations` or `reject`) with a written justification for
every criterion.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20research%20prototype-orange)

---

## Why this is not just "ask an LLM to rate a paper"

A raw LLM judgment on a paper is confident, unverifiable, and wrong in ways you cannot
see. Four design choices address that directly:

| Problem | What the pipeline does |
|---|---|
| The model cannot know if an institution or an author is real | Author affiliations are resolved against **ROR** and **OpenAlex**, author track records against **Semantic Scholar**, by DOI first, with name-matching only as a labelled fallback |
| The model invents supporting quotes | Every quote it cites as evidence is **matched back against the extracted PDF text** (`validate_quotes`); an unverifiable quote caps that criterion's score |
| The model bluffs about the bibliography | Reference entries are resolved against **Crossref**; the `references` score is computed **deterministically** from the verified count, not from the model's opinion |
| Silent data gaps become silent scoring errors | Enrichment failures are surfaced as explicit `data_quality_warnings` in the output, and the prompt tells the model which fields it may *not* trust |

Everything the model is asked to do that it can't be trusted to do alone is either
grounded in an API or checked after the fact.

---

## Architecture

```mermaid
flowchart TD
    A[Preprint PDF] --> B[pdfminer extraction<br/>header + full text]

    B --> C[LLM pass 1<br/>extract authors & affiliations]
    C --> D{External grounding}
    D --> D1[ROR<br/>institution identity + aliases]
    D --> D2[OpenAlex<br/>affiliations by DOI, field match]
    D --> D3[Semantic Scholar<br/>h-index, publication record]

    B --> E[LLM pass 2<br/>content criteria 2-4<br/>native PDF input]
    D --> F[LLM pass 3<br/>criterion 1: author credibility]

    B --> G[Crossref<br/>reference verification<br/>no LLM]
    A --> H[PREreview<br/>community feedback<br/>criterion 5]

    E --> I[Quote validation<br/>against source text]
    F --> I
    G --> J[Deterministic<br/>reference score]

    I --> K[Score aggregation]
    J --> K
    H --> K
    K --> L[JSON result + summary CSV<br/>accept / with reservations / reject]
```

The three LLM calls are deliberately separate: author credibility is scored from
structured, API-verified data only, so a persuasive paper cannot talk its way into a
better affiliation score.

---

## The rubric

Nine scored rows, grouped into five criteria. Each row is scored 1 (weak) / 2 (moderate) /
3 (strong) on its own evidence, and composite criteria average their rows rather than
requiring every condition to hold at once. The whole thing is declared in
[`criteria.yaml`](criteria.yaml), so the scoring logic lives in data rather than in prompt
strings.

| # | Criterion | Row | What earns a 3 | Grounded in |
|---|---|---|---|---|
| 1 | **Author credibility** | institution reputability | All authors at credible institutions **and** the corresponding author at a nationally or globally recognised one | ROR + OpenAlex |
| | | author expertise | 3 or more authors with a verifiable publication record in the field | Semantic Scholar |
| | | institutional collaboration | More than 3 institutes involved | OpenAlex |
| 2 | **Research question & methods** | objective & hypothesis | Objective and hypotheses explicit, testable, aligned with prior evidence | Paper text (quote-validated) |
| | | public-health relevance | Addresses a pressing concern **and** relates to a known outbreak (e.g. WHO DON) | Paper text (quote-validated) |
| | | study-design rigour | Rigorous design, methods validating the results, data and materials available for replication | Paper text (quote-validated) |
| 3 | **Results & conclusion** | | Findings internally consistent, validated by multiple methods, limitations explicitly discussed | Paper text (quote-validated) |
| 4 | **References** | | Balanced mix of foundational, recent and domain-relevant peer-reviewed work (more than ~20) | Crossref |
| 5 | **Community feedback** | | Multiple expert reviews, follow-up analyses or citations, authors responding to feedback | PREreview |

<details>
<summary><strong>Full 1 / 2 / 3 definitions for every row</strong></summary>

**1a. Institution reputability**: *Authors are affiliated with reputable universities, research institutes or recognised labs*
1. Some authors affiliated with low-credibility institutions such as predatory "universities" or commercial entities.
2. All authors affiliated with a credible university or lab with some research output and relevant expertise.
3. The above, and the main or corresponding author is from a well-established, globally or nationally recognised research institution.

**1b. Author expertise**: *Authors have verifiable expertise and prior publications in the field*
1. Little or none.
2. Up to 2 authors with verifiable expertise.
3. Three or more authors with verifiable expertise.

**1c. Institutional collaboration**: *The study is collaborative, involving multiple institutes*
1. Single institution.
2. Two or three institutes.
3. More than three institutes.

**2a. Objective and hypothesis**: *The study objective and hypotheses are clearly written*
1. Objective is unclear, missing or vague.
2. Objective is clear but lacks precision or a clear connection to methods and results.
3. Objective and hypotheses are explicit, testable and aligned with theory or prior evidence, with a clear sense of why the study matters.

**2b. Public-health relevance**: *The study addresses a pressing public health concern*
1. No.
2. Yes.
3. Yes, and it relates to a known outbreak (e.g. WHO Disease Outbreak News).

**2c. Study-design rigour**: *Study design is clear and rigorous, with sufficient detail for reproducibility*
1. Methodology is missing.
2. Design briefly described but lacking detail, or preliminary results with no validation.
3. Clear and rigorous design with methods validating the results; data, code and materials openly available or described in enough detail to replicate.

**3. Results and conclusion**
1. Results lack supporting figures or tables, or are inconsistent with the data and methods described; conclusions overstated, speculative or not grounded in the evidence.
2. Some inconsistencies in reporting or unclear statistical parameters; conclusions cautious but lacking clarity or only partially supported.
3. Findings internally consistent, validated by multiple methods where applicable, aligned with the hypotheses; results clearly interpreted and limitations explicitly discussed.

**4. References**
1. Fewer than ~10 references, missing foundational studies; overreliance on non-peer-reviewed sources, obscure journals or self-citations.
2. Relevant and credible sources but omitting some important recent work; roughly 10 to 20 references.
3. Balanced mix of foundational, recent and domain-relevant peer-reviewed studies (more than ~20), used to justify methods, contextualise findings and discuss limitations.

**5. Community feedback**
1. No comments or reviews on the preprint server; mentions limited to social media or unverified sources.
2. Feedback on clarifications or methods but not full validation; early signs of citation or scholarly discussion; mixed or tentative reception.
3. Multiple expert-level comments or peer reviews; follow-up analyses or scholarly citations already exist; authors have responded responsibly.

</details>

Two rows are not scored by the model, by design. **References** is computed
deterministically from the verified reference count rather than from the model's opinion
of the bibliography, because the model kept misreading its own count against the
threshold. **Community feedback** is scored only when PREreview actually holds reviews
for the DOI; otherwise it stays unscored and drops out of the average, rather than
scoring a 1 that would apply to almost every preprint.

An earlier design required every condition in a criterion to hold at once for a 3. It
made that score nearly unreachable and was replaced by the row-averaging above after the
first real runs.

---

## Quickstart

```bash
git clone https://github.com/JuanFinello/preprints-classification-system.git
cd preprints-classification-system
pip install -r requirements.txt
cp .env.example .env    # then add your API key
export OPENAI_API_KEY="sk-..."
```

Score a single preprint:

```bash
python evaluate_preprint.py \
    --pdf papers/10.1101_2025.06.14.659623.pdf \
    --criteria criteria.yaml \
    --out results_pipeline/10.1101_2025.06.14.659623.json
```

Score a directory of preprints and consolidate into one CSV:

```bash
python batch_evaluate.py \
    --papers-dir papers \
    --results-dir results_pipeline \
    --summary results_summary_pipeline.csv
```

Cross-model calibration (optional, needs `ANTHROPIC_API_KEY`):

```bash
python batch_evaluate_claude.py   # scores the same papers with Claude
python compare_scores.py          # agreement matrix between the two models
```

### Output shape

```json
{
  "source_pdf": "10.1101_2025.06.14.659623.pdf",
  "model": "gpt-5.6-terra",
  "n_pages": 60,
  "author_enrichment": { "...ROR / OpenAlex / Semantic Scholar evidence..." },
  "criteria": {
    "author_credibility": { "score": 3, "justification": "...", "sub_criteria": {} },
    "research_question_and_methods": { "score": 3, "quote_valid": true },
    "results_and_conclusion": { "score": 2, "justification": "..." },
    "references": { "score": 3, "citation_verification": {} },
    "feedback": { "score": null }
  },
  "scoring": {
    "average": 2.75, "total": 11, "max": 12,
    "recommendation": "accept", "quotes_invalid": []
  },
  "data_quality_warnings": []
}
```

Real examples live in [`results_pipeline/`](results_pipeline/); the flattened view of all
runs is in [`results_summary_pipeline.csv`](results_summary_pipeline.csv).

---

## What this project learned the hard way

These are the findings that shaped the design; they are documented because they are the
useful part.

1. **Content-policy refusals are an architectural constraint, not an edge case.**
   Molecular-virology preprints (viral receptors, host restriction factors,
   protein-host interactions) were refused outright by one provider's API. The refusal
   reproduced across four isolated diagnostics, including an 8 000-character abstract
   with a trivial instruction, which ruled out prompt length and rubric wording. Surveillance
   and diagnostics papers passed cleanly. For a genomic-surveillance use case that
   pattern hits exactly the papers that matter most, so the pipeline is built
   multi-backend (`evaluate_preprint.py`, `_claude.py`, `_deepseek.py` share one rubric
   and one scoring core) rather than betting on one vendor.

2. **Anchoring is real and criterion-specific.** In an early run, one model returned
   `results_and_conclusion = 1` for four papers out of four while the other model
   spread across 1 to 3. Verified in code that no deterministic function was forcing it:
   the anchoring was pure prompt bias, and it was fixed in the prompt, not in the
   aggregation.

3. **An agreement percentage can be meaningless.** The `references` criterion agreed
   ~65 % of the time between models, but one side computes it from a Crossref count and
   the other makes a qualitative judgment. Agreement there measures coincidence, not
   consensus. Comparisons are only worth reporting when both sides answer the same
   question.

4. **A failed run must never look like a low score.** A parse failure was written to
   disk as `score: 1, justification: "Parse error"`, which aggregates to a perfectly
   plausible `reject`. Failures need to be structurally distinguishable from judgments.

---

## Repository layout

| Path | Role |
|---|---|
| `evaluate_preprint.py` | Core pipeline: extraction, API grounding, prompts, quote validation, scoring |
| `criteria.yaml` | The rubric: criteria, sub-criteria and score definitions |
| `batch_evaluate.py` · `weekly_update.py` | Batch runs over a directory of PDFs, and the weekly orchestration around them |
| `evaluate_preprint_claude.py` · `_deepseek.py` · `compare_scores.py` | Alternative model backends and the cross-model agreement report |
| `results_pipeline/` · `results_summary_pipeline.csv` | Structured outputs of real runs |

Source PDFs and internal curation data (DOI worklists, curator spreadsheets) are
deliberately not versioned; see `.gitignore`.

---

## Known limitations

- **Thresholds are provisional.** The `accept` / `reject` cut-offs in `_recommendation()`
  are informed guesses pending a larger labelled set.
- **Figure-based evidence cannot be quote-validated.** When the model reasons from a
  figure, `validate_quotes` has no text to match against; those findings need a human
  look.
- **Scores do not yet feed back into the curation decision.** Today the output informs a
  human curator; wiring it into the upstream worklist is the next step.

---

## Roadmap

- [ ] Distinguish failures from judgments across the whole write path
- [ ] Prompt work on `results_and_conclusion`, the criterion where the two models agree least
- [ ] Stamp a rubric version into every stored result
- [ ] Complete the DeepSeek backend and re-run the refusal comparison across all three
- [ ] Feed recommendations back into the curator worklist

---

## License

MIT. See [LICENSE](LICENSE).

Built for genomic-surveillance curation work. Preprint content belongs to its respective
authors; this repository contains only automatically generated assessments of publicly
available preprints.
