#!/usr/bin/env python3

import argparse
import os
import subprocess
from lxml import html
from pathlib import Path
import requests

def get_file_links(base_url):
    response = requests.get(base_url, timeout=30)
    if response.status_code != 200:
        print(f"Failed to fetch data from {base_url}")
        return []

    page = response.text
    tree = html.fromstring(page)
    file_links = [base_url + link for link in tree.xpath('//a[contains(@href, ".xml.gz") and not(contains(@href, ".md5"))]/@href')]
    return file_links

def download_files(download_dir, file_links):
    """ Download each file from a list of file URLs """
    for file_url in file_links:
        file_name = file_url.split('/')[-1]
        local_path = os.path.join(download_dir, file_name)
        # check if file already downloaded 
        if not os.path.exists(local_path):
            print(f"Downloading: {file_url}")
            result = subprocess.call(['wget', '-P', download_dir, file_url])
            if result != 0:
                print(f"Download failed with exit code {result}: {file_url}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=Path, required=True, help="Path to pubmed download directory")
    parser.add_argument("--update_mode", action="store_true", help="Only download the latest publications from Pubmed")
    args = parser.parse_args()

    download_dir = os.path.join(args.home_dir, 'raw_download_files')
    os.makedirs(download_dir, exist_ok=True)

    # URLs for PubMed baseline and update files
    PUBMED_BASELINE = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
    PUBMED_UPDATE = 'https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/'

    baseline_files = get_file_links(PUBMED_BASELINE)
    update_files = get_file_links(PUBMED_UPDATE)

    if args.update_mode:
        all_files = update_files
    else:
        all_files = baseline_files + update_files

    download_files(download_dir, all_files)

if __name__ == "__main__":
        main()
