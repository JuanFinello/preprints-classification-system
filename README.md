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
| The model bluffs about the bibliography | The `references` score is computed **in code**, never by the model: the count it reports sets the base and the quality problems it reports can only lower it. Reference entries are separately resolved against **Crossref**, recorded for a human to read |
| Silent data gaps become silent scoring errors | Enrichment failures are surfaced as explicit `data_quality_warnings` in the output, and the prompt tells the model which fields it may *not* trust |

Everything the model is asked to do that it can't be trusted to do alone is either
grounded in an API or checked after the fact.

The honest exception is reference quality. Whether a cited work is off-topic,
non-peer-reviewed or a self-citation is the model's own reading of the bibliography, and
nothing verifies it: those three inputs lower the `references` score on the model's word
alone. Two of them could be grounded, since Crossref returns a work's type and author list for
every entry that resolves, and the reference-list parser now reaches 93 % of entries
(up from 52 %) so those counts would no longer be drawn from a biased slice of the
bibliography. Until that grounding is built, the caps are explicitly a model judgment,
recorded in `score_note` so a reviewer can see exactly which entries triggered them.

---

## Architecture

```mermaid
flowchart TD
    A[Preprint PDF] --> B[pdfminer extraction<br/>header + full text]

    A --> DOI([DOI])
    B --> C[LLM 1<br/>read authors & affiliations<br/>off the header]

    DOI --> D2[OpenAlex<br/>institutions by DOI<br/>primary source]
    DOI --> D3[Semantic Scholar<br/>publication record, h-index]
    C --> D3
    C -. fallback: only when OpenAlex<br/>has no record for the DOI .-> D1[ROR<br/>affiliation strings<br/>author link lost, warned]

    D2 --> F[LLM 2<br/>criterion 1: author credibility<br/>structured data only, no paper text]
    D1 --> F
    D3 --> F

    A --> E[LLM 3<br/>criteria 2-4<br/>native PDF: text + page images]
    E --> I[Quote validation<br/>every cited quote matched<br/>against the extracted text]
    E --> J[References scored in code<br/>count sets the base<br/>quality findings cap it]

    A --> H[PREreview<br/>reviews for this DOI]
    H --> G[LLM 4<br/>criterion 5<br/>only when reviews exist]

    B --> V[Crossref<br/>do the cited works resolve?<br/>no LLM, informational only]

    F --> K[Score aggregation]
    I --> K
    J --> K
    G --> K
    K --> L[JSON result + summary CSV<br/>accept / with reservations / reject<br/>or error if a call produced nothing]
    V -.-> L
```

The calls are deliberately separate. Author credibility never sees the paper text, only
the structured data the APIs returned, so a persuasive paper cannot talk its way into a
better affiliation score. Note which way the arrows run: the institution lookup is
anchored on the DOI, not on what the model read off the header, so a misread affiliation
cannot poison the grounding. The model's extraction is used only when OpenAlex has no
record for the DOI, and that path is recorded as a data-quality warning because it loses
the author-to-institution link. The Crossref check runs alongside the score rather than into
it: it reports whether each cited work resolves to a real indexed publication, and is
recorded in the output for a human to read.

---

## The rubric

Nine scored rows, grouped into five criteria. Each row is scored 1 (weak) / 2 (moderate) /
3 (strong) on its own evidence, and composite criteria average their rows rather than
requiring every condition to hold at once. The whole thing is declared in
[`criteria.yaml`](criteria.yaml), so the scoring logic lives in data rather than in prompt
strings.

| Criterion | Row | A 3 requires | Grounded in |
|---|---|---|---|
| **1. Author credibility** | institution reputability | corresponding author at a recognised institution | ROR + OpenAlex |
| | author expertise | 3+ authors with a publication record in the field | Semantic Scholar |
| | institutional collaboration | more than 3 institutes | OpenAlex |
| **2. Research question & methods** | objective & hypothesis | explicit, testable, tied to prior evidence | paper text |
| | public-health relevance | tied to a known outbreak | paper text |
| | study-design rigour | validated methods, data available to replicate | paper text |
| **3. Results & conclusion** | | consistent, multi-method, limitations discussed | paper text |
| **4. References** | | 20+, balanced, domain-relevant, peer-reviewed | count + quality, scored in code |
| **5. Community feedback** | | expert reviews, citations, author responses | PREreview |

Everything in the paper-text rows is quote-validated: the model has to cite the sentence
it scored on, and the sentence has to be in the paper.

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

**References** is the one row the model never scores itself: it reports facts and the
code applies the rule. The count sets the base, and the quality findings can only pull it
down. Any entry flagged as genuinely off-topic caps the score at 2, and a bibliography
more than half non-peer-reviewed or self-cited drops to 1. The split exists because the
model could judge quality but could not reliably compare its own count against a
threshold, once calling 58 references "within 10-20". Crossref separately checks that the
cited works resolve to real indexed publications, recorded in the output and kept out of
the score.

**Community feedback** is scored only when PREreview actually holds reviews for the DOI;
otherwise it stays unscored and drops out of the average, rather than scoring a 1 that
would apply to almost every preprint.

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
   ~65 % of the time between two models, but one side derived it from a reference count
   and the other made a qualitative judgment. Agreement there measured coincidence, not
   consensus. Comparisons are only worth reporting when both sides answer the same
   question, which is part of why that criterion now scores quality explicitly rather
   than counting alone.

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
