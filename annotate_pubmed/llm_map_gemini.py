#!/usr/bin/env python3
import argparse
import ast
import csv
import configparser
import os
import sys
from urllib.request import urlopen
from enum import Enum
from html.parser import HTMLParser
from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account
from pydantic import BaseModel
import json
import re
from pathlib import Path
from urllib.error import HTTPError
import xml.etree.ElementTree as ET


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def _parse_top5_cross(raw_value):
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return [x for x in raw_value if isinstance(x, dict)]
    if isinstance(raw_value, dict):
        return [raw_value]

    text = str(raw_value).strip()
    if not text:
        return []

    # Try JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback for python-style literals and malformed separators.
    normalized = re.sub(r"\s+", " ", text)
    if not (normalized.startswith("[") and normalized.endswith("]")):
        normalized = f"[{normalized}]"
    normalized = re.sub(r"}\s*{", "}, {", normalized)
    try:
        parsed = ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def _format_g2p_record(label: str) -> dict:
    if not label or label == "":
        return {}
    parts = label.split("---")
    return {
        "g2p_id": parts[0].strip(),
        "gene_symbol": parts[1].strip(),
        "old_gene_symbol": parts[4].strip(),
        "disease": parts[5].strip(),
        "genotype": parts[8].strip(),
        "confidence": parts[10].strip(),
        "mechanism": parts[13].strip(),
    }


def load_json_key(key_file):
    credentials = service_account.Credentials.from_service_account_file(key_file).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
    return credentials


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
                title = root.findtext(".//article-title")
                abstract = " ".join(p.text.strip() for p in root.findall(".//abstract/p") if p.text)
                sections = []
                for sec in root.findall(".//body/*"):
                    sec_title = sec.findtext("title")
                    paragraphs = []

                    # Some section have the text directly there
                    for p in sec.findall("p"):
                        text = get_text_clean(p)
                        paragraphs.append(text.replace("\n", " "))
                    
                    # Other sections have sub-sections inside
                    for sub_section in sec.findall("sec"):
                        for p in sub_section.findall("p"):
                            sub_section_text = get_text_clean(p)
                            paragraphs.append(sub_section_text.replace("\n", " "))

                    sections.append((sec_title, paragraphs))

                full_text = ""
                for sec_title, paras in sections:
                    if sec_title:
                        full_text += "\n"+sec_title+"\n"
                    for p in paras:
                        full_text += " "+p
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

def get_text_clean(element):
    """
    Extract all text inside an element.
    Keep text from <italic> and <bold>.
    Skip text inside <xref>, but preserve its tail text.
    """
    parts = []

    # Add leading text for the current element (if any)
    if element.text:
        parts.append(element.text)

    # Recursively process children
    for child in element:
        if child.tag == "xref":
            # Skip xref.text, but keep any tail text
            if child.tail:
                parts.append(child.tail)
        else:
            # Keep text and tail for all other tags (italic, bold, etc.)
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


class Label(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DISPUTED = "disputed"


class Relevance(BaseModel):
    label: Label
    comment: str


def process_publication(
    client: genai.Client, record: dict, article: dict, model: str
) -> Relevance:
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
        prompt += "\nMolecular mechanism: "+record['mechanism']

    if article['fulltext']:
        prompt += "\nFull text: "+article['fulltext']

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
    focus_columns = ["pmid", "title", "abstract", "top5_cross"]

    parser = argparse.ArgumentParser(
        description="Convert parquet to CSV using pmid, title, abstract, and expanded top5_cross objects."
    )
    parser.add_argument(
        "--parquet_file",
        type=Path,
        required=True,
        help="Path to parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gemini_output.csv"),
        help="Output CSV file (default: gemini_output.csv)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV by appending new rows and skipping already processed PMID+G2P pairs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file with Google Vertex AI settings(key file, project name, model, etc.)",
    )
    args = parser.parse_args()

    # Read config file
    config_parser = configparser.ConfigParser()
    config_parser.read(args.config)
    config = config_parser["project_config"]

    if "key_file" not in config or "project" not in config:
        sys.exit("Error: 'key_file' or 'project' not found in config file")

    credentials = load_json_key(config["key_file"])

    if "location" not in config:
        config["location"] = "europe-west2" # default location

    if "model" not in config:
        config["model"] = "gemini-2.5-flash" # default model

    client = genai.Client(
        vertexai=True,
        project=config["project"],
        location=config["location"], # gemini pro is available at us-central1; flash is in europe-west2
        credentials=credentials,
        http_options=HttpOptions(api_version="v1")
    )

    parquet_path = args.parquet_file.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        raise RuntimeError(
            "No parquet reader found. Install one with:\n"
            "  pip install pyarrow\n"
            "Then run this script again."
        )

    parquet_file = pq.ParquetFile(parquet_path)
    print(f"File: {parquet_path}")
    print(f"Rows: {parquet_file.metadata.num_rows}")

    available_columns = set(parquet_file.schema_arrow.names)
    missing_columns = [c for c in focus_columns if c not in available_columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")

    object_fields = set()
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=focus_columns)
        df = table.to_pandas()
        for _, row in df.iterrows():
            objects = _parse_top5_cross(row["top5_cross"])
            for obj in objects:
                object_fields.update(str(k) for k in obj.keys())

    object_field_list = sorted(object_fields)
    header = ["pmid", "g2p_id", "relevance_label", "relevance_comment"]

    rows_analysed = []
    processed_pairs = set()

    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            for existing_row in reader:
                pmid = _clean_text(existing_row.get("pmid", ""))
                g2p_id = _clean_text(existing_row.get("g2p_id", ""))
                if pmid and g2p_id:
                    processed_pairs.add((pmid, g2p_id))
        print(f"Resume mode: found {len(processed_pairs)} previously processed PMID+G2P pairs in {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(file_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if file_mode == "w":
            writer.writeheader()

        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=focus_columns)
            df = table.to_pandas()
            for _, row in df.iterrows():
                base = {
                    "pmid": _clean_text(row["pmid"]),
                    "title": _clean_text(row["title"]),
                    "abstract": _clean_text(row["abstract"]),
                }

                objects = _parse_top5_cross(row["top5_cross"])
                if not objects:
                    out = dict(base)
                    for field in object_field_list:
                        out[field] = ""
                    continue

                for obj in objects:
                    out = dict(base)
                    for field in object_field_list:
                        value = obj.get(field, "")
                        if isinstance(value, (dict, list)):
                            out[field] = json.dumps(value, ensure_ascii=False)
                        else:
                            out[field] = _clean_text(value)
                    # Filter by score before saving row to be analysed by Gemini
                    if float(out.get("score", 0)) >= 0.1:
                        # Format the G2P record
                        g2p_record = _format_g2p_record(out.get("label", ""))
                        out.update(g2p_record)
                        pair = (_clean_text(out.get("pmid", "")), _clean_text(out.get("g2p_id", "")))
                        if args.resume and pair in processed_pairs:
                            continue

                        try:
                            article = get_article(out["pmid"])
                            if article:
                                relevance = process_publication(client, g2p_record, article, config["model"])
                                out["relevance_label"] = relevance.label.value
                                out["relevance_comment"] = relevance.comment

                                print(out["pmid"], out["g2p_id"], out["relevance_label"], out["relevance_comment"])
                                writer.writerow({k: out.get(k, "") for k in header})
                                f.flush()
                                os.fsync(f.fileno())
                                if args.resume:
                                    processed_pairs.add(pair)
                        except Exception as e:
                            print(f"Error processing PMID {out.get('pmid', '')} G2P {out.get('g2p_id', '')}: {e}", file=sys.stderr)

                        rows_analysed.append(out)

    print(f"Wrote CSV: {output_path}")
    print(f"top5_cross object fields: {len(object_field_list)}")
    if object_field_list:
        print("Fields: " + ", ".join(object_field_list))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
