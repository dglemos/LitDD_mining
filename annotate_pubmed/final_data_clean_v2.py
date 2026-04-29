#!/usr/bin/env python3
"""
This script is a reimplementation of the final_data_clean.py script.

Description:
It reads the parquet output file from the LLM run, filters out invalid results,
and can optionally run a Gemini relevance analysis for each valid PMID/G2P pair.
It can also resume from an existing output CSV.
Note: Gemini analysis can be slow, so it is recommended to use it together
with --resume.

Usage:
  python final_data_clean_v2.py \
    --llm_file pubmed_bert_positive_2025_crossencoded_shard0-of-1__llm.parquet \
    --g2p_file g2p_2025.csv \
    --gene2pubtator gene2pubtator.gz \
    --output_csv final_cleaned_data.csv \
    [--gemini_config gemini_config.ini] \
    [--resume]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.request import urlopen

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a parquet file and print a summary.")
    parser.add_argument(
        "--llm_file",
        required=True,
        help="Output parquet file of the LLM step",
    )
    parser.add_argument(
        "--g2p_file",
        required=True,
        help="G2P CSV file",
    )
    parser.add_argument(
        "--gene2pubtator",
        required=True,
        help="gene2pubtator compressed file",
    )
    parser.add_argument(
        "--score_cutoff",
        type=float,
        default=0.9,
        help="Minimum top5_cross score to keep a row",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV file with PMID and G2P_IDs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug printing",
    )
    parser.add_argument(
        "--gemini_config",
        default=None,
        help="Optional Gemini config file. If provided, run Gemini analysis for each valid PMID/G2P pair.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV by appending new rows and skipping already processed PMID/G2P pairs.",
    )
    return parser.parse_args()


def load_json_key(key_file):
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(key_file).with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )
    return credentials


def get_text_clean(element):
    """Extract all text inside an element while skipping xref text."""
    parts = []

    if element.text:
        parts.append(element.text)

    for child in element:
        if child.tag == "xref":
            if child.tail:
                parts.append(child.tail)
        else:
            parts.append(get_text_clean(child))
            if child.tail:
                parts.append(child.tail)

    return "".join(parts)


class PlainTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks = []

    def handle_data(self, data):
        if data.strip():
            self.chunks.append(data.strip())

    def get_text(self):
        return " ".join(self.chunks)


def get_article(pmid: int) -> dict | None:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/article/MED/{pmid}?resultType=core&format=json"
    with urlopen(url) as res:
        data = json.loads(res.read())

    if data["hitCount"] == 0:
        return None

    obj = data["result"]

    full_text = None
    full_text_list = obj.get("fullTextIdList")
    if full_text_list:
        full_text_id = full_text_list.get("fullTextId")[0]

        url_full_text = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{full_text_id}/fullTextXML"
        try:
            with urlopen(url_full_text) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                sections = []
                for sec in root.findall(".//body/*"):
                    sec_title = sec.findtext("title")
                    paragraphs = []

                    for p in sec.findall("p"):
                        text = get_text_clean(p)
                        paragraphs.append(text.replace("\n", " "))

                    for sub_section in sec.findall("sec"):
                        for p in sub_section.findall("p"):
                            sub_section_text = get_text_clean(p)
                            paragraphs.append(sub_section_text.replace("\n", " "))

                    sections.append((sec_title, paragraphs))

                full_text = ""
                for sec_title, paras in sections:
                    if sec_title:
                        full_text += "\n" + sec_title + "\n"
                    for p in paras:
                        full_text += " " + p
        except HTTPError as e:
            print(f"Error {e.code}: Full text not found for PMID {pmid}")

    title = obj.get("title")
    abstract = obj.get("abstractText")
    if not title or not abstract:
        return None

    parser = PlainTextExtractor()
    parser.feed(abstract)

    if "journalInfo" not in data["result"]:
        journal_info = "Unknown"
    else:
        journal_info = data["result"]["journalInfo"]["journal"]["title"]

    return {
        "title": title,
        "abstract": parser.get_text(),
        "journal": journal_info,
        "fulltext": full_text,
    }


def process_publication(client, record: dict, article: dict, model: str):
    from enum import Enum
    from pydantic import BaseModel

    class Label(Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        DISPUTED = "disputed"

    class Relevance(BaseModel):
        label: Label
        comment: str

    prompt = f"""\
You are assessing whether a scientific publication is relevant \
to a Gene2Phenotype record defined by a gene and a disease.
Relevance means the publication provides or discusses evidence \
that this gene is causally or mechanistically linked to this \
disease in humans or relevant models (e.g. mammalian or functional \
models recapitulating the human phenotype).

Output one of four labels:
- high: the article directly supports or reports an association \
between the specified gene and the specified disease (same disease, \
not just related systems).
- medium: the article discusses the specified gene or disease in a \
mechanistically relevant or closely related context, but without \
demonstrating a direct association between the two. \
The disease context should still be similar \
(e.g. same phenotype family or allelic requirement) and the article should provide \
new evidence for the gene-disease association.
- low: the article discusses the gene or disease in an unrelated context, \
or links the gene to a different disease than the one specified or focuses on \
a different gene.
- disputed: The article provides evidence that contradicts or disproves \
an association between the specified gene and the specified disease.

Then provide one short reason.

NEVER assign "high" relevance unless \
the publication provides evidence directly linking the gene to the \
specific disease named in the record.
If the article discusses a different disease caused by the same gene: \
- If the diseases share overlapping molecular mechanisms and phenotypes, assign "medium"; \
- If they do not, assign "low".
Consider whether the molecular mechanism described in the publication \
(e.g. gain or loss of function) matches the mechanism in the record (if available) \
when assessing similarity.
If the publication or the record do not mention a molecular mechanism, base your \
decision on the gene-disease association itself (e.g. clinical or genetic evidence). \
Do not lower relevance solely because the mechanism is unspecified.
If the publication discusses multiple genes or structural variants involving \
the specified gene then assign "low" relevance.

Input:
Gene: {record['gene_symbol']}
Previous gene symbols: {record['old_gene_symbol']}
Disease: {record['disease']}
Title: {article['title']}
Abstract: {article['abstract']}\
"""
    if "mechanism" in record:
        prompt += "\nMolecular mechanism: " + record["mechanism"]

    if article["fulltext"]:
        prompt += "\nFull text: " + article["fulltext"]

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": Relevance,
            "temperature": 0.2,
        },
    )

    return response.parsed


def main() -> int:
    args = parse_args()
    parquet_path = args.llm_file
    g2p_path = args.g2p_file
    gene2pubtator_path = args.gene2pubtator
    score_cutoff = args.score_cutoff
    output_csv = args.output_csv
    debug = args.debug
    gemini_config = args.gemini_config
    resume = args.resume

    total_count= 0
    valid_count = 0
    gemini_enabled = gemini_config is not None
    gemini_client = None
    gemini_model = None

    if gemini_enabled:
        try:
            import configparser
            from google import genai
            from google.genai.types import HttpOptions
        except ModuleNotFoundError as exc:
            print(f"Gemini dependencies are missing: {exc}", file=sys.stderr)
            return 1

        config_path = Path(gemini_config).expanduser().resolve()
        if not config_path.exists():
            print(f"Gemini config file not found: {config_path}", file=sys.stderr)
            return 1

        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        if "project_config" not in config_parser:
            print("Gemini config is missing [project_config]", file=sys.stderr)
            return 1
        gemini_project_config = config_parser["project_config"]

        if "key_file" not in gemini_project_config or "project" not in gemini_project_config:
            print("Gemini config requires 'key_file' and 'project'", file=sys.stderr)
            return 1

        credentials = load_json_key(gemini_project_config["key_file"])
        gemini_location = gemini_project_config.get("location", "europe-west2")
        gemini_model = gemini_project_config.get("model", "gemini-2.5-flash")
        gemini_client = genai.Client(
            vertexai=True,
            project=gemini_project_config["project"],
            location=gemini_location,
            credentials=credentials,
            http_options=HttpOptions(api_version="v1"),
        )

    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}", file=sys.stderr)
        return 1
    if not os.path.exists(g2p_path):
        print(f"File not found: {g2p_path}", file=sys.stderr)
        return 1
    if not os.path.exists(gene2pubtator_path):
        print(f"File not found: {gene2pubtator_path}", file=sys.stderr)
        return 1

    # Build G2P map: g2p id -> [gene symbol, previous gene symbols...]
    g2p_map_gene = {}
    g2p_records = {}
    with open(g2p_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g2p_id = (row.get("g2p id") or "").strip()
            if not g2p_id:
                continue
            gene_symbol = (row.get("gene symbol") or "").strip()
            prev = (row.get("previous gene symbols") or "").strip()
            prev_list = [p.strip() for p in prev.split(";") if p.strip()]
            values = []
            if gene_symbol:
                values.append(gene_symbol)
            values.extend(prev_list)
            g2p_map_gene[g2p_id] = values
            g2p_records[g2p_id] = {
                "g2p_id": g2p_id,
                "gene_symbol": gene_symbol,
                "old_gene_symbol": prev,
                "disease": (row.get("disease name") or "").strip(),
                "genotype": (row.get("allelic requirement") or "").strip(),
                "confidence": (row.get("confidence category") or "").strip(),
                "mechanism": (row.get("mutation consequence") or "").strip(),
            }

    # Stream through parquet and build dicts: pmid -> llm_dis_map, pmid -> top5_cross
    pmid_to_llm = {}
    pmid_to_top5 = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["pmid", "llm_dis_map", "top5_cross"]):
        pmids = batch.column(0).to_pylist()
        llm_vals = batch.column(1).to_pylist()
        top5_vals = batch.column(2).to_pylist()
        for pmid, llm_val, top5_val in zip(pmids, llm_vals, top5_vals):
            pmid_to_llm[pmid] = llm_val
            # Build dict with scores: g2p id -> score
            for item in top5_val:
                label = item.get("label", "")
                score = item.get("score", None)
                # Extract the g2p id from the label
                m = re.search(r"^(G2P\d+)\b", label or "")
                g2p_id = m.group(1) if m else None
                if pmid not in pmid_to_top5:
                    pmid_to_top5[pmid] = {g2p_id: score}
                else:
                    pmid_to_top5[pmid][g2p_id] = score

    pmid_set = set(pmid_to_llm.keys())

    # Build gene2pubtator map: first column -> list of fourth column values.
    gene2pubtator_map = {}
    with gzip.open(gene2pubtator_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            key = parts[0]
            if key not in pmid_set:
                continue
            value = parts[3]
            values = [v.strip() for v in value.split("|") if v.strip()]
            gene2pubtator_map.setdefault(key, []).extend(values)

    processed_pairs = set()
    header = ["PMID", "G2P_IDs"]
    if gemini_enabled:
        header.extend(["gemini_relevance_label", "gemini_relevance_comment"])

    output_path = Path(output_csv)
    if resume and output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as existing_f:
            reader = csv.DictReader(existing_f)
            for row in reader:
                pmid = str(row.get("PMID", "")).strip()
                g2p_id = str(row.get("G2P_IDs", "")).strip()
                if pmid and g2p_id:
                    processed_pairs.add((pmid, g2p_id))

    file_mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(file_mode, newline="", encoding="utf-8") as out_f:
        out_writer = csv.writer(out_f)
        if file_mode == "w":
            out_writer.writerow(header)

        for pmid in pmid_to_llm:
            if pmid_to_llm[pmid] == "NO MATCH":
                continue
            list_of_g2ps = pmid_to_llm[pmid].split(";")
            list_of_genes_for_pmid = gene2pubtator_map.get(pmid, [])
            genes_for_pmid_set = set(list_of_genes_for_pmid)

            if debug:
                print(f"\nPMID: {pmid}; genes: {list_of_genes_for_pmid}")

            for g2p in list_of_g2ps:
                total_count += 1
                g2p = g2p.strip()

                # Filter by top5_cross score
                if score_cutoff:
                    top5_scores = pmid_to_top5.get(pmid, {})
                    score = top5_scores.get(g2p, None)
                    if score is not None and score < score_cutoff:
                        if debug:
                            print(f"{pmid}\t{g2p}\tSCORE BELOW CUTOFF ({score:.2f} < {score_cutoff})")
                        continue

                if g2p not in g2p_map_gene:
                    continue

                g2p_genes = g2p_map_gene[g2p]
                if any(g in genes_for_pmid_set for g in g2p_genes):
                    pair = (str(pmid).strip(), str(g2p).strip())
                    if resume and pair in processed_pairs:
                        if debug:
                            print(f"{pmid}\t{g2p}\tSKIP RESUME")
                        continue
                    valid_count += 1
                    if debug:
                        print(f"{pmid}\t{g2p}\tVALID")
                    row = [pmid, g2p]
                    if gemini_enabled:
                        relevance_label = ""
                        relevance_comment = ""
                        try:
                            article = get_article(pmid)
                            if article:
                                relevance = process_publication(
                                    gemini_client,
                                    g2p_records.get(g2p, {}),
                                    article,
                                    gemini_model,
                                )
                                relevance_label = relevance.label.value
                                relevance_comment = relevance.comment
                            elif debug:
                                print(f"{pmid}\t{g2p}\tGemini skipped: article not found")
                        except Exception as exc:
                            print(
                                f"Gemini error for PMID {pmid} G2P {g2p}: {exc}",
                                file=sys.stderr,
                            )
                        row.extend([relevance_label, relevance_comment])
                    out_writer.writerow(row)
                    if resume:
                        processed_pairs.add(pair)
                else:
                    continue

    print(f"Loaded: {parquet_path}")
    print(f"Total mappings: {total_count}")
    print(f"Valid mappings: {valid_count}")
    print(f"Wrote: {output_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
