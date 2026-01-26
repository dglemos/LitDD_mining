#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from glob import glob
from pathlib import Path
import pubmed_parser as pp
import traceback

def process_file_to_parquet(xml_file, output_directory):
    base_name = os.path.splitext(os.path.splitext(os.path.basename(xml_file))[0])[0]
    output_file = os.path.join(output_directory, f'{base_name}.parquet')

    if os.path.exists(output_file):
        print(f"Skipping {xml_file}, as {output_file} already exists.")
        return

    print(f"Processing {xml_file}")
    try:
        docs = pp.parse_medline_xml(xml_file, year_info_only=False)
        docs_list = list(docs)
        df = pd.DataFrame(docs_list)
        pub_year = df['pubdate'].astype(str).str.split('-').str[0]
        df['pubdate'] = pd.to_numeric(pub_year, errors='coerce')
        df.to_parquet(output_file, engine='pyarrow', index=False)
        print(f"Saved {output_file}")

    except Exception as e:
        error_info = str(e) + "\n" + traceback.format_exc()
        with open(f'BAD_DOWNLOAD_{base_name}.txt', 'w') as f:
            f.write(error_info)
        print(f"Error processing {xml_file}, logged to BAD_DOWNLOAD_{base_name}.txt")

def get_xml_files(directory):
    for xml_file in glob(os.path.join(directory, '*.xml.gz')):
        yield xml_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_dir", type=Path, required=True, help="Directory to raw download files")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory to parquet download files")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    xml_files_generator = get_xml_files(args.download_dir)

    for xml_file in xml_files_generator:
        process_file_to_parquet(xml_file, args.output_dir)

if __name__ == "__main__":
        main()
