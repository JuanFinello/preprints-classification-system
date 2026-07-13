#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detecta los registros marcados como preprint en daily_dois.xlsx,
consulta Crossref para ver si tienen DOI publicado asociado,
y exporta los resultados a CSV.

INPUT:
    daily_dois.xlsx

OUTPUT:
    preprints_checked_crossref.csv
    preprints_with_published_doi.csv
"""

import time
import argparse
from pathlib import Path

import pandas as pd
import requests


CROSSREF = "https://api.crossref.org/works/{}"
USER_AGENT = "JuanFinello-GISAID-preprint-check/1.0 (mailto:juanfinello@gmail.com)"


def extract_related_dois(cr_item):
    """
    Extrae DOI relacionados desde el campo 'relation' de Crossref.
    En particular busca 'is-preprint-of', que suele indicar el DOI publicado.
    """
    rel = (cr_item or {}).get("relation") or {}
    out = {"is_preprint_of": []}

    vals = rel.get("is-preprint-of") or []
    for v in vals:
        if isinstance(v, dict):
            rid = v.get("id") or v.get("DOI") or v.get("doi")
            if rid:
                out["is_preprint_of"].append(str(rid).lower().strip())
        elif isinstance(v, str):
            out["is_preprint_of"].append(v.lower().strip())

    # Dejar valores únicos preservando el orden
    seen = set()
    out["is_preprint_of"] = [
        x for x in out["is_preprint_of"]
        if x and not (x in seen or seen.add(x))
    ]

    return out


def classify_doi(doi, session):
    """
    Clasifica un DOI consultando Crossref.

    Posibles estados:
        published: parece tener DOI publicado asociado o ya es artículo publicado
        preprint: Crossref lo marca como preprint / posted-content
        unknown: no hay señal suficiente
    """
    doi_norm = str(doi).lower().strip()
    url = CROSSREF.format(requests.utils.quote(doi_norm, safe=""))

    try:
        r = session.get(url, timeout=30)

        if r.status_code == 404:
            return {
                "doi": doi_norm,
                "state": "unknown",
                "published_doi": "",
                "reason": "404"
            }

        r.raise_for_status()
        item = r.json().get("message", {})

    except Exception as e:
        return {
            "doi": doi_norm,
            "state": "unknown",
            "published_doi": "",
            "reason": type(e).__name__
        }

    cr_type = (item.get("type") or "").lower()
    related = extract_related_dois(item)

    # 1. Señal explícita: este DOI es preprint de otro DOI publicado
    if related["is_preprint_of"]:
        return {
            "doi": doi_norm,
            "state": "published",
            "published_doi": related["is_preprint_of"][0],
            "reason": "relation_is_preprint_of"
        }

    # 2. Tipo Crossref: artículo publicado
    if cr_type in {"journal-article", "proceedings-article"}:
        return {
            "doi": doi_norm,
            "state": "published",
            "published_doi": doi_norm,
            "reason": "crossref_type_published"
        }

    # 3. Tipo Crossref: preprint
    if cr_type in {"posted-content", "preprint"}:
        return {
            "doi": doi_norm,
            "state": "preprint",
            "published_doi": "",
            "reason": "crossref_type_preprint"
        }

    return {
        "doi": doi_norm,
        "state": "unknown",
        "published_doi": "",
        "reason": "no_signal"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check preprint DOIs from daily_dois.xlsx against Crossref."
    )
    parser.add_argument(
        "-i", "--input",
        default="daily_dois.xlsx",
        help="Input Excel file. Default: daily_dois.xlsx"
    )
    parser.add_argument(
        "--all-output",
        default="preprints_checked_crossref.csv",
        help="CSV output with all checked preprints."
    )
    parser.add_argument(
        "--published-output",
        default="preprints_with_published_doi.csv",
        help="CSV output only with preprints that have a published DOI."
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to wait between Crossref requests. Default: 0.2"
    )
    args = parser.parse_args()

    daily_dois_file = Path(args.input)

    if not daily_dois_file.exists():
        raise FileNotFoundError(f"No encontré el archivo: {daily_dois_file}")

    # ---- leer daily_dois ----
    df_daily = pd.read_excel(daily_dois_file, dtype=str)

    # ---- validación mínima ----
    required_cols = {"DOI", "Status"}
    missing = required_cols - set(df_daily.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {daily_dois_file}: {missing}. "
            f"Columnas disponibles: {list(df_daily.columns)}"
        )

    # ---- normalizar columnas clave ----
    df_daily["DOI"] = df_daily["DOI"].astype(str).str.strip().str.lower()
    df_daily["Status"] = df_daily["Status"].astype(str).str.strip().str.lower()

    # ---- filtrar preprints ----
    df_preprints = df_daily[
        df_daily["Status"].str.contains("preprint", na=False)
    ].copy()

    print(f"Preprints detectados: {len(df_preprints)}")

    if df_preprints.empty:
        print("No se detectaron preprints. No se generaron archivos de salida.")
        return

    # ---- quedarnos solo con columnas relevantes ----
    df_preprints = df_preprints[["DOI", "Status"]].copy()

    # ---- lista de DOIs únicos ----
    dois = (
        df_preprints["DOI"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("nan")]
        .drop_duplicates()
        .tolist()
    )

    print(f"DOIs únicos a consultar en Crossref: {len(dois)}")
    print(f"Primeros 5 DOIs: {dois[:5]}")

    # ---- sesión Crossref ----
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # ---- consultar Crossref ----
    results = []

    for i, doi in enumerate(dois, 1):
        results.append(classify_doi(doi, session))

        if i % 25 == 0 or i == len(dois):
            print(f"Consultados: {i}/{len(dois)}")

        time.sleep(args.sleep)

    res_df = pd.DataFrame(results)

    print("\nResumen de estados:")
    print(res_df["state"].value_counts(dropna=False))

    # ---- unir resultados a df_preprints ----
    df_preprints["_doi_norm"] = df_preprints["DOI"].astype(str).str.lower().str.strip()
    res_df["_doi_norm"] = res_df["doi"].astype(str).str.lower().str.strip()

    df_preprints_checked = df_preprints.merge(
        res_df.drop(columns=["doi"]),
        on="_doi_norm",
        how="left"
    ).drop(columns=["_doi_norm"])

    # ---- filtrar preprints con DOI publicado ----
    df_pub = df_preprints_checked[
        df_preprints_checked["published_doi"].notna()
        & (df_preprints_checked["published_doi"].astype(str).str.strip() != "")
    ].copy()

    # ---- guardar outputs ----
    df_preprints_checked.to_csv(
        args.all_output,
        sep=",",
        index=False,
        encoding="utf-8"
    )

    df_pub.to_csv(
        args.published_output,
        sep=",",
        index=False,
        encoding="utf-8"
    )

    print(f"\nArchivo con todos los preprints chequeados: {args.all_output}")
    print(f"Archivo con preprints que tienen DOI publicado: {args.published_output}")
    print(f"Preprints con published_doi detectados: {len(df_pub)}")


if __name__ == "__main__":
    main()
