PDF ──▶ RAG 1 ──▶ evaluación científica
                 │
                 ▼
        APIs externas (ROR, PubMed, preprint)
                 │
                 ▼
               RAG 2 ──▶ contexto / prioridad



python extract_text_from_pdf.py --pdf preprint.pdf --out paper_text.json
python split_sections.py --in paper_text.json --out paper_sections.json
python retriever_part1.py --sections paper_sections.json --criteria criteria.yaml --out retrieved_context.json
